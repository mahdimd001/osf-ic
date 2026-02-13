# 402 , 448 , 329
import numpy as np
import torch
import time
import opentuner
from opentuner import ConfigurationManipulator, EnumParameter, MeasurementInterface, Result
from transformers import AutoModelForImageClassification
from torch.utils.data import DataLoader
from osf import OFM 
import datetime
from datasets import load_dataset,Image as DatasetImage
from transformers import AutoImageProcessor, AutoModelForImageClassification
from itertools import combinations
import functools
import sys
from huggingface_hub import login
from PIL import Image
from collections import defaultdict



arc_sample = {'layer_1': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_2': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_3': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_4': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_5': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_6': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_7': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_8': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_9': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_10': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_11': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}, 'layer_12': {'atten_out': 768, 'inter_hidden': 960, 'residual_hidden': 648}}

def create_architecture(cfg, arc_sample):
    """
    Updates arc_sample in-place with values from cfg.
    """
    # 1. Retrieve the global residual value
    global_residual = cfg.get('residual_hidden_space')

    # 2. Iterate through every layer in the target dictionary
    for layer_name, layer_data in arc_sample.items():
        
        # Update residual_hidden for ALL layers
        if global_residual is not None:
            layer_data['residual_hidden'] = global_residual

        # Update inter_hidden specific to this layer
        # We construct the key (e.g., "layer_1" + "_inter_hidden")
        cfg_key = f"{layer_name}_inter_hidden"
        
        if cfg_key in cfg:
            layer_data['inter_hidden'] = cfg[cfg_key]
            
    return arc_sample



train_dataloader = None  # Placeholder: You must define this based on your dataset
val_dataloader = None  # Placeholder: You must define this based on your dataset
reorder_dataloader = None  # Placeholder: You must define this based on your dataset

ofm = None  # Placeholder: You must define this based on your OFM model
supermodel = None  # Placeholder: You must define this based on your supermodel
file_name = None
max_model_counter = 0

model = "/lustre/hdd/LAS/jannesar-lab/msamani/DEIT_Cifar10_Cont/second_22/f1_best_model"
dataset_name = 'cifar10'
huggingface_token = ""
cache_dir = "/lustre/hdd/LAS/jannesar-lab/msamani/.cache"
labels = None
processor_name = "google/vit-base-patch16-224"
batch_size = 128


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def transform(example_batch, processor):
        try:
            inputs = processor([x.convert("RGB") for x in example_batch["img"]], return_tensors="pt")
            inputs["labels"] = example_batch["label"]
            return inputs
        except Exception as e:
            print(f"Error processing batch: {e}")
            my_logger.warning(f"Error processing batch: {e}")
            # Handle the case where images cannot be processed
            # You can return a dummy input or skip this batch
            # For now, we will return an empty tensor
            inputs = processor([Image.new("RGB", (224, 224))], return_tensors="pt")
            inputs["labels"] = torch.tensor([-1])
            return inputs

def load_model_dataset():
    
    """Load the model and dataset based on provided arguments."""
    global train_dataloader, val_dataloader, model, ofm, supermodel, file_name, dataset_name, reorder_dataloader, labels, processor_name, batch_size




    # Authenticate for ImageNet-1k
    if dataset_name == "imagenet-1k" and huggingface_token:
        login(token=huggingface_token)
        
    login(token=huggingface_token)
    # Load dataset
    print("loading dataset...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir,trust_remote_code=True)


    if dataset_name == "cifar100":
        dataset = dataset.rename_column("fine_label", "label")
        train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
    elif dataset_name == "imagenet-1k":
        # ImageNet-1k already has train/validation splits
        if "validation" not in dataset:
            train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
            dataset["train"] = train_val["train"]
            dataset["validation"] = train_val["test"]
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['validation'].rename_column("image", "img")

    elif dataset_name == "cifar10":
        train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
    elif dataset_name == "slegroux/tiny-imagenet-200-clean":
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['validation'].rename_column("image", "img")
    elif dataset_name == "zh-plus/tiny-imagenet":
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['valid'].rename_column("image", "img")
    else:
        print(f"Dataset {dataset_name} not supported. Please choose from ['imagenet-1k', 'cifar100', 'cifar10', 'slegroux/tiny-imagenet-200-clean', 'zh-plus/tiny-imagenet']")
        sys.exit(1)


    





    labels = dataset["train"].features["label"].names
    print("loading processor...")
    # Initialize processor and dataset
    processor = AutoImageProcessor.from_pretrained(processor_name, cache_dir=cache_dir, use_fast=True)
    #prepared_ds = dataset.with_transform(functools.partial(transform, processor=processor))

    


    prepared_ds = {
    "train": dataset["train"].with_transform(functools.partial(transform, processor=processor)),
    "validation": dataset["validation"].with_transform(functools.partial(transform, processor=processor)),
    }
    print("train loader...")
    # Create data loaders
    train_dataloader = DataLoader(
        prepared_ds["train"], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True, drop_last=True
    )
    print("test loader...")
    val_dataloader = DataLoader(
        prepared_ds["validation"], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True, drop_last=False
    )





    print("loading pretrained model...")
    # Initialize model
    model = AutoModelForImageClassification.from_pretrained(
        model,  #args.model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_safetensors=True
    )


    ofm = OFM(model.to("cpu"))

    



# Define the search space creation function
def arc_config_creator(
    inter_hidden_space: list[int],
    residual_hidden_space: list[int],
) -> dict:
    """Generate search space for ViT architecture configuration.

    Args:
        inter_hidden_space (list[int]): List of possible intermediate FFN hidden sizes.
        residual_hidden_space (list[int]): List of possible residual hidden sizes.

    Returns:
        dict: Search space for OpenTuner.
    """
    search_space = {
        "residual_hidden_space": residual_hidden_space,
    }


    for layer in range(1, 12): 
        conf_key = f"layer_{layer}_inter_hidden"
        search_space[conf_key] = inter_hidden_space
    return search_space

# Define the search space
search_space = arc_config_creator(
    inter_hidden_space=[3072,1280,960], 
    residual_hidden_space=[ 768, 648],     
)

def summarize_config(config_dict):
    log_parts = []

    # inter_hidden per layer
    inter_hidden_parts = []
    for i in range(12):
        if str(i) in config_dict:
            inter_hidden = config_dict[str(i)].get('inter_hidden', 'N/A')
            inter_hidden_parts.append(f"L{i}:{inter_hidden}")
    log_parts.append("inter_hidden=[" + ",".join(inter_hidden_parts) + "]")

    # removed layer indices
    removed_layers = config_dict.get('remove_layer_idx', [])
    log_parts.append("removed_layers=" + str(removed_layers))

    # heads to prune
    heads_to_prune = config_dict.get('heads_config', {}).get('heads_to_prune', 'N/A')
    log_parts.append(f"heads_to_prune={heads_to_prune}")

    # removed_heads (non-empty only)
    removed_heads = config_dict.get('removed_heads', {})
    head_parts = []
    for k, v in removed_heads.items():
        if v and v != [...]:  # skip empty or placeholder lists
            head_parts.append(f"L{k}:{v}")
    if head_parts:
        log_parts.append("removed_heads={" + ",".join(head_parts) + "}")

    return " | ".join(log_parts)



# Evaluation function for classification accuracy
def eval(subnetwork, val_dataloader, device='cuda'):
    """Evaluate the subnetwork on a validation set and return classification accuracy."""
    subnetwork.eval()
    subnetwork = subnetwork.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch['pixel_values'].to(device)  # Adjust key based on your dataset
            labels = batch['labels'].to(device)        # Adjust key based on your dataset
            outputs = subnetwork(pixel_values=images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy

class NASOpenTuner(MeasurementInterface):
    def manipulator(self):
        """Define the configuration manipulator for OpenTuner."""
        manipulator = ConfigurationManipulator()
        for key in search_space:
            manipulator.add_parameter(EnumParameter(key, search_space[key]))
        return manipulator

    def run(self, desired_result, input, limit):
        
        """Compile and run a given configuration, then return performance."""
        cfg = desired_result.configuration.data


        
        
        

        # Create the architecture configuration
        arc_config = {}
        # Add layer-specific configurations for retained layers

        arc_config = create_architecture(cfg, arc_sample)
        

        # Generate the subnetwork
        subnetwork, total_params = ofm.resource_aware_model(arc_config)





        # Evaluate the subnetwork
        accuracy = eval(subnetwork, val_dataloader, device=device)
        # smart_accuracy = eval(smart_subnetwork, val_dataloader, device=device)

        # if total_params != smart_total_params:
        #     f.write(f"Warning: Total parameters mismatch! Original: {total_params}, Smart: {smart_total_params}\n")
        global max_model_counter
        max_model_counter += 1
        if max_model_counter >= 300:  # Limit the number of models to tune
            sys.exit(0)
            
        print(f"Model {max_model_counter}: Accuracy={accuracy:.4f}, Total Params={total_params}")
        metriic = (1 - accuracy) + (total_params / ofm.total_params) * 0.1

        return Result(time=(metriic))  # Minimize error (1 - accuracy)

   
if __name__ == '__main__':
    load_model_dataset()
    argparser = opentuner.default_argparser()
    start = time.time()
    NASOpenTuner.main(argparser.parse_args())
    stop = time.time()
    print(f"Tuning took {stop - start:.2f} seconds")