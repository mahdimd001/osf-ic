import os
import torch
import numpy as np
from datasets import load_dataset
import functools
import evaluate
from transformers import AutoImageProcessor, AutoModelForImageClassification
from arguments import arguments
from osf import OFM
from osf.utils.trainer import TrainingArguments, Trainer
from torch.utils.data import DataLoader, Subset



def compute_metrics(eval_pred):
    """This function is used to compute the metrics for the evaluation.

    Args:
        eval_pred: The output of the Trainer.evaluate function
    returns:
        A dictionary of metrics
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accuracy = accuracy_metric.compute(
        predictions=np.argmax(eval_pred["predictions"], axis=1),
        references=eval_pred["label_ids"],
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(eval_pred["predictions"], axis=1),
        references=eval_pred["label_ids"],
        average="weighted",
    )

    return {"metric": accuracy["accuracy"], "f1": f1["f1"]}


def collate_fn(batch):
    """This function is used to collate the data samples into batches.
    It is used to supply the DataLoader with the collate_fn argument.

    Args:
        batch: A list of samples from the dataset
    returns:
        A dictionary of tensors containing the batched samples
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def transform(example_batch, processor):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor(
        [x.convert("RGB") for x in example_batch["img"]], return_tensors="pt"
    )

    # Include the labels
    inputs["labels"] = example_batch["label"]
    return inputs


def main(args):
    if args.model == "vit":
        model_name = "google/vit-base-patch16-224"
        processor_name = "google/vit-base-patch16-224"
    elif args.model == "vit-large":
        model_name = "google/vit-large-patch16-224-in21k"
        processor_name = "google/vit-large-patch16-224"
    elif args.model == "swinv2":
        model_name = "microsoft/swin-base-patch4-window7-224-in22k"
        processor_name = "microsoft/swin-base-patch4-window7-224"  # pre-trained
    elif args.model == "deit":
        model_name = "facebook/deit-base-patch16-224"
        processor_name = "facebook/deit-base-patch16-224"
    elif args.model == "osf":
        model_name = "anonymous-429/osf-vit-base-patch16-224-imagenet"
        processor_name = "google/vit-base-patch16-224"
    # load data and preprocess

    if args.huggingface_token:
        from huggingface_hub import login

        login(args.huggingface_token)

    dataset = load_dataset(
        args.dataset, cache_dir=args.cache_dir, trust_remote_code=True
    )

    if args.dataset == "imagenet-1k":
        assert (
            args.huggingface_token is not None
        ), "Please provide a HuggingFace token to download the ImageNet dataset"
        dataset = dataset.rename_column("image", "img")

    if args.dataset in ["cifar100", "cifar10"]:
        if args.dataset == "cifar100":
            dataset = dataset.rename_column("fine_label", "label")

        train_val = dataset["train"].train_test_split(
            test_size=0.2, stratify_by_column="label", seed=123
        )
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]

    labels = dataset["train"].features["label"].names

    processor = AutoImageProcessor.from_pretrained(
        processor_name, cache_dir=args.cache_dir, use_fast=True
    )
    prepared_ds = dataset.with_transform(
        functools.partial(transform, processor=processor)
    )

    # load/initialize global model and convert to raffm model
    if args.resume_ckpt:
        ckpt_path = args.resume_ckpt
        elastic_config = (
            os.path.join(ckpt_path, "elastic_space.json")
            if os.path.exists(os.path.join(ckpt_path, "elastic_space.json"))
            else args.elastic_config
        )

    else:
        ckpt_path = model_name
        elastic_config = args.elastic_config

    model = AutoModelForImageClassification.from_pretrained(
        ckpt_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
        use_safetensors=True,

    )

    # Reordering dataset
    # Create a subset and apply the same transform as the main dataset
    reordering_subset = dataset["train"].select(range(0, 300 * args.batch_size, 1)).with_transform(
        functools.partial(transform, processor=processor)
    )
    reorder_dataloader = DataLoader(
        reordering_subset, 
        batch_size=args.batch_size,  # Match main batch size for consistency
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=False, 
        drop_last=False
    )
    args.reorder_dataloader = reorder_dataloader

    model = OFM(model.to("cpu"), elastic_config)

    trainer = Trainer(
        model,
        TrainingArguments(
            output_dir=args.save_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            report_to=[],
            fp16=args.fp16,
            dataloader_num_workers=8,
            log_interval=args.log_interval,
            reorder = args.reorder,
            reorder_method = args.reorder_method,
            reorder_dataloader = args.reorder_dataloader,
            smart_medium = args.smart_medium
        ),
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        test_dataset=prepared_ds["test"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        optimizers=(None, None),
    )
    metrics = trainer.train()

    model.save_ckpt(os.path.join(args.save_dir, "final"))


if __name__ == "__main__":
    args = arguments()
    # write args to file
    # create save_dir if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    main(args)
