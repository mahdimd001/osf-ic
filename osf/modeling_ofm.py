""" Official Implementation
One Foundation Model Fits All: Single-stage Foundation Model Training with Zero-shot Deployment
"""

import copy
import os
from typing import Any
import pandas as pd
import torch
from torch import nn
from .model_downsize import (
    bert_module_handler,
    arc_config_sampler,
    vit_module_handler,
    sam_module_handler,
    t5_module_handler,
    roberta_module_handler,
    distilbert_module_handler,
    swin_module_handler,
    mamba_module_handler,
    clip_module_handler,
)
from .param_prioritization import *
from .utils import calculate_params, save_dict_to_file, load_dict_from_file
from .weight_reorder import vit_weight_reorder


def update_arch_config(arc_config, sample_series):
    """
    Updates arc_config in-place using values from sample_series.
    """
    # Map the prefix in the Series index to the key in the nested dictionary
    key_mapping = {
        'attention': 'atten_out',
        'inter_hidden': 'inter_hidden',
        'residual': 'residual_hidden'
    }

    for key, value in sample_series.items():
        # 1. Split the key to separate the prefix from the layer number
        # rsplit('_', 1) splits from the right, ensuring 'inter_hidden_1' splits into 'inter_hidden' and '1'
        try:
            prefix, layer_num = key.rsplit('_', 1)
        except ValueError:
            continue # Skip keys that don't match the format

        # 2. Update the dictionary if the prefix is valid
        if prefix in key_mapping:
            layer_name = f"layer_{layer_num}"
            inner_key = key_mapping[prefix]
            
            # Check if layer exists to avoid errors
            if layer_name in arc_config:
                # Use .item() to convert numpy/pandas types to native Python types (int/float)
                # This ensures the dict is JSON serializable later if needed
                arc_config[layer_name][inner_key] = int(value.item()) if hasattr(value, 'item') else int(value)

    return arc_config


class OFM:
    def __init__(self, model, elastic_config=None) -> None:
        self.model = model
        self.total_params = calculate_params(model=model)

        if hasattr(self.model.config, "elastic_config"):
            elastic_config = self.model.config.elastic_config

        if not elastic_config:
            # set defalt search space configuration (this is defalt setting for bert)
            elastic_config = {
                "atten_out_space": [768],
                "inter_hidden_space": [3072, 1920, 1280],
                "residual_hidden_space": [768],
            }
            print(
                f"[Warning]: No elastic configuration provides. Set to the defalt elastic space {elastic_config}."
            )
        elif isinstance(elastic_config, str):
            elastic_config = load_dict_from_file(elastic_config)

        assert isinstance(
            elastic_config, dict
        ), "Invalid elastic_config, expect input a dictionary or file path"

        self.model.config.elastic_config = elastic_config
        # self.elastic_config = elastic_config
        self.local_grads = []
        self.alphas = []
        self._pre_global_grad = None

    def random_resource_aware_model(self):
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        if "sam" == self.model.config.model_type.lower():
            arc_config = arc_config_sampler(
                **self.model.config.elastic_config,
                n_layer=self.model.vision_encoder.config.num_hidden_layers,
            )
        elif "swin" == self.model.config.model_type.lower():
            arc_config = arc_config_sampler(
                **self.model.config.elastic_config,
                n_layer=self.model.config.depths[-2],
            )

        elif "clip" == self.model.config.model_type.lower():
            text_arc_config = arc_config_sampler(
                **self.model.config.elastic_config["text"],
                n_layer=self.model.config.text_config.num_hidden_layers,
            )
            vision_arc_config = arc_config_sampler(
                **self.model.config.elastic_config["vision"],
                smallest=True,
                n_layer=self.model.config.vision_config.num_hidden_layers,
            )
            arc_config = (text_arc_config, vision_arc_config)
        else:
            arc_config = arc_config_sampler(
                **self.model.config.elastic_config,
                n_layer=self.model.config.num_hidden_layers,
            )

        subnetwork, total_params = self.resource_aware_model(arc_config)

        return subnetwork, total_params, arc_config
    
    def smart_medium_model(self,sample_arch_config=None):
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        
        arc_config = arc_config_sampler(
            **self.model.config.elastic_config,
            n_layer=self.model.config.num_hidden_layers,
        )
        arc_config = update_arch_config(arc_config, sample_arch_config)
        
        

        subnetwork, total_params = self.resource_aware_model(arc_config)

        return subnetwork, total_params, arc_config
    

    def mlp_layer_reordering(self,dataloader=None,method='magnitude'):
        if "vit" == self.model.config.model_type.lower():
            self.model, score_dist = vit_weight_reorder(self.model,dataloader,method)
            return score_dist
        else:
            raise NotImplemented(f'Weight reordering not yet implemented for \
                                 {self.model.config.model_type.lower()}')

    def smallest_model(self):
        """Return the smallest model in the elastic space

        Returns:
            - subnetwork (nn.Module): The smallest model in the elastic space
            - params (int): The number of parameters in million of the smallest model
            - arc_config (dict): The configuration of the smallest model
        """

        if "sam" == self.model.config.model_type.lower():
            arc_config = arc_config_sampler(
                **self.model.config.elastic_config,
                smallest=True,
                n_layer=self.model.vision_encoder.config.num_hidden_layers,
            )
        elif "swin" == self.model.config.model_type.lower():
            arc_config = arc_config_sampler(
                **self.model.config.elastic_config,
                smallest=True,
                n_layer=self.model.config.depths[-2],
            )

        elif "clip" == self.model.config.model_type.lower():
            text_arc_config = arc_config_sampler(
                **self.model.config.elastic_config["text"],
                n_layer=self.model.config.text_config.num_hidden_layers,
                smallest=True,
            )
            vision_arc_config = arc_config_sampler(
                **self.model.config.elastic_config["vision"],
                smallest=True,
                n_layer=self.model.config.vision_config.num_hidden_layers,
            )
            arc_config = (text_arc_config, vision_arc_config)

        else:
            arc_config = arc_config_sampler(
                **self.model.config.elastic_config,
                smallest=True,
                n_layer=self.model.config.num_hidden_layers,
            )
        subnetwork, params = self.resource_aware_model(arc_config)
        return subnetwork, params, arc_config

    def largest_model(self):
        return copy.deepcopy(self.model), self.total_params, {}

    def resource_aware_model(self, arc_config):
        if "bert" == self.model.config.model_type.lower():
            return bert_module_handler(self.model, arc_config)
        elif "vit" == self.model.config.model_type.lower():
            return vit_module_handler(self.model, arc_config)
        elif "sam" == self.model.config.model_type.lower():
            return sam_module_handler(self.model, arc_config)
        elif "t5" == self.model.config.model_type.lower():
            return t5_module_handler(self.model, arc_config)
        elif "roberta" == self.model.config.model_type.lower():
            return roberta_module_handler(self.model, arc_config)
        elif "distilbert" == self.model.config.model_type.lower():
            return distilbert_module_handler(self.model, arc_config)
        elif "swin" == self.model.config.model_type.lower():
            return swin_module_handler(self.model, arc_config)
        elif "mamba" == self.model.config.model_type.lower():
            return mamba_module_handler(self.model, arc_config)
        elif "clip" == self.model.config.model_type.lower():
            return clip_module_handler(self.model, arc_config)
        else:
            raise NotImplementedError

    def salient_parameter_prioritization(self, metric=l1_norm):
        self.model = salient_parameter_prioritization(self.model, metric)

    def grad_accumulate(self, local_grad, alpha=None):
        self.local_grads.append(local_grad)
        self.alphas.append(alpha)

    def apply_grad(self, grad):
        """Apply the gradients to the full-size model

        Args:
            grad (dict): Trained downsized model gradients
        """
        self.model.to("cpu")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                local_grad = grad[name].cpu()
                slices = tuple(
                    slice(0, min(sm_dim, lg_dim))
                    for sm_dim, lg_dim in zip(local_grad.shape, param.shape)
                )
                if self._pre_global_grad:
                    param[slice] -= (
                        0.9 * local_grad + 0.1 * self._pre_global_grad[name][slice]
                    )
                else:
                    param[slices] -= local_grad

    def apply_accumulate_grad(self, beta=0.5):
        self.grad_normalization()

        self.model.to("cpu")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                for local_grad, alpha in zip(self.local_grads, self.alphas):
                    local_param_grad = local_grad[name].cpu()
                    slices = tuple(
                        slice(0, min(sm_dim, lg_dim))
                        for sm_dim, lg_dim in zip(local_param_grad.shape, param.shape)
                    )
                    param[slices] -= (
                        local_param_grad * alpha / sum(self.alphas)
                    ) * beta

        self.local_grads.clear()
        self.alphas.clear()

    def train(
        self,
        args,
        data_shards,
        val_dataset,
        test_dataset=None,
        processor=None,
        collate_fn=None,
        compute_metrics=None,
    ):
        pass

    def grad_normalization(self):
        """Normalize the gradients via previous epoch's gradients"""
        pass

    def save_ckpt(self, dir):
        self.model.save_pretrained(os.path.join(dir))

    def load_ckpt(self, dir):
        self.model = self.model.from_pretrained(dir)
        # check the the existance of self.model.config.elastic_config
        assert hasattr(
            self.model.config, "elastic_config"
        ), "No elastic configuration found in the model config file. Please check the config file."
