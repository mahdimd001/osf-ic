import copy
import os
import time
import numpy as np
from ..utils import EarlyStopping, Logger
from ..modeling_ofm import OFM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import networkx as nx
from ..my_forman_ricci import FormanRicciGPU

def extract_hidden_lists(layers_dict):
    atten_out = []
    inter_hidden = []
    residual_hidden = []

    # Sort by layer number to ensure correct order
    for layer in sorted(layers_dict.keys(), key=lambda x: int(x.split('_')[1])):
        atten_out.append(layers_dict[layer]['atten_out'])
        inter_hidden.append(layers_dict[layer]['inter_hidden'])
        residual_hidden.append(layers_dict[layer]['residual_hidden'])

    return atten_out, inter_hidden, residual_hidden


wanda_matrices = {}

def wanda_matrix_hook(inst, inp, out, layer, lin):
    """
    Captures the full Wanda Matrix (|W| * ||A||) for the layer.
    """
    with torch.no_grad():
        # W shape: [Out, In]
        W = inst.weight
        C_out = W.shape[1]

        # Calculate L2 norm of input activations across batch and tokens
        # inp[0] is [Batch, Tokens, In_Dim] -> Flatten to [-1, In_Dim]
        l2_norm = inp[0].reshape(-1, C_out).norm(p=2, dim=0) # [In_Dim]
        
        # Wanda Matrix: Broadcast multiply
        # [Out, In] * [In] -> [Out, In]
        wanda_mat = W.abs() * l2_norm
        
        # Accumulate
        if lin == 1:
            if 'W1' not in wanda_matrices[layer]:
                wanda_matrices[layer]['W1'] = torch.zeros_like(wanda_mat)
            wanda_matrices[layer]['W1'] += wanda_mat
        elif lin == 2:
            if 'W2' not in wanda_matrices[layer]:
                wanda_matrices[layer]['W2'] = torch.zeros_like(wanda_mat)
            wanda_matrices[layer]['W2'] += wanda_mat
            
        # Count batches (increment only once per layer pass, using lin=1)
        if lin == 1:
            wanda_matrices[layer]['count'] = wanda_matrices[layer].get('count', 0) + 1



def vit_wanda_graph_ricci_compute(model, dataloader, sparse_threshold=0.01):
    """
    1. Compute Wanda Matrices (Activation-Aware Weights).
    2. Build Graph using Wanda Matrices (Traffic-based Topology).
    3. Compute Ricci Curvature on this Traffic Graph.
    4. Reorder based on Hybrid Score (Wanda_Mag - Ricci).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # --- PHASE 1: Calibration (Compute Wanda Matrices) ---
    print("Phase 1: Building Activation-Aware Graphs (Wanda Calibration)...")
    
    global wanda_matrices
    wanda_matrices = {i: {} for i in range(len(model.vit.encoder.layer))}
    
    hooks = []
    for idx, layer in enumerate(model.vit.encoder.layer):
        h1 = layer.intermediate.dense.register_forward_hook(partial(wanda_matrix_hook, layer=idx, lin=1))
        h2 = layer.output.dense.register_forward_hook(partial(wanda_matrix_hook, layer=idx, lin=2))
        hooks.extend([h1, h2])
        
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Forward Passes"):
            if isinstance(batch, dict):
                inputs = batch.get("pixel_values", batch.get("img")).to(device)
            else:
                inputs = batch[0].to(device)
            model(inputs)
            
    for h in hooks: h.remove()
    
    # --- PHASE 2 & 3: Geometric Analysis & Reordering ---
    print("Phase 2: Computing Forman-Ricci on Wanda Graphs...")
    
    score_dist = {}
    
    for i, layer in enumerate(tqdm(model.vit.encoder.layer, desc="Reordering")):
        # Retrieve averaged Wanda Matrices
        count = wanda_matrices[i]['count']
        # W1_wanda: [Hidden, Input]
        W1_wanda = (wanda_matrices[i]['W1'] / count).cpu().numpy()
        # W2_wanda: [Output, Hidden]
        W2_wanda = (wanda_matrices[i]['W2'] / count).cpu().numpy()
        
        D_hid, D_in = W1_wanda.shape
        D_out, _ = W2_wanda.shape
        
        # --- Build Graph based on WANDA SCORES ---
        G = nx.DiGraph()
        
        # Thresholding based on Wanda strength
        thresh_w1 = sparse_threshold * np.max(W1_wanda)
        thresh_w2 = sparse_threshold * np.max(W2_wanda)
        
        # Add Edges: Distance is inverse of Wanda Score
        # (High Traffic = Short Distance = Strong Connection)
        rows, cols = np.where(W1_wanda >= thresh_w1)
        for r, c in zip(rows, cols):
            w_val = W1_wanda[r, c]
            G.add_edge(c, D_in + r, weight=1.0/(w_val + 1e-6), layer='W1', hidden_idx=r)
            
        rows, cols = np.where(W2_wanda >= thresh_w2)
        for r, c in zip(rows, cols):
            w_val = W2_wanda[r, c]
            G.add_edge(D_in + c, D_in + D_hid + r, weight=1.0/(w_val + 1e-6), layer='W2', hidden_idx=c)
            
        # --- Compute Ricci ---
        ricci_neuron_scores = np.zeros(D_hid)
        
        if G.number_of_edges() > 0:
            try:
                # Using Forman-Ricci for speed
                frc = FormanRicciGPU(G)
                frc.compute_ricci_curvature()
                
                for u, v, data in frc.G.edges(data=True):
                    if 'formanCurvature' in data:
                        h_idx = data.get('hidden_idx')
                        if h_idx is not None:
                            ricci_neuron_scores[h_idx] += data['formanCurvature']
            except Exception as e:
                print(f"Ricci Error Layer {i}: {e}")
        
        # --- Compute Node-Level Wanda Magnitude ---
        # (Total traffic passing through the neuron)
        w1_sum = np.sum(W1_wanda, axis=1) # [Hidden]
        w2_sum = np.sum(W2_wanda, axis=0) # [Hidden]
        wanda_mag = (w1_sum + w2_sum) / 2.0
        
        # --- Hybrid Combination ---
        # Normalize
        w_norm = (wanda_mag - wanda_mag.min()) / (wanda_mag.max() - wanda_mag.min() + 1e-6)
        
        max_r = np.max(np.abs(ricci_neuron_scores))
        if max_r > 0:
            r_norm = ricci_neuron_scores / max_r
        else:
            r_norm = ricci_neuron_scores
            
        # Score = Traffic_Strength - Traffic_Bottleneck
        # (Negative Ricci on Wanda Graph = Critical Traffic Bottleneck)
        hybrid_scores = - r_norm
        
        score_dist[i] = hybrid_scores.tolist()
        
        # # --- Reorder ---
        # sorted_indices = np.argsort(hybrid_scores)[::-1]
        # idx = torch.from_numpy(sorted_indices.copy()).long().to(device)
        
        # # Apply to actual weights
        # layer.intermediate.dense.weight.data = layer.intermediate.dense.weight.data[idx, :]
        # layer.output.dense.weight.data = layer.output.dense.weight.data[:, idx]
        # layer.intermediate.dense.bias.data = layer.intermediate.dense.bias.data[idx]
        
    return score_dist


def sum_scores(scores, hidden_list):
    total_list = []
    mean_list = []
    var_list = []
    for i, h in enumerate(hidden_list):
        # If hidden_list[i] is 0 → sum = 0
        if h <= 0:
            total_list.append(0)
            mean_list.append(0)
            var_list.append(0)
            continue
        
        # Safely sum scores[i] up to h
        total = sum(scores.get(i, [])[:h])
        mean = total / h if h > 0 else 0
        var = sum((x - mean) ** 2 for x in scores.get(i, [])[:h]) / h if h > 0 else 0

        total_list.append(total)
        mean_list.append(mean)
        var_list.append(var)
    
    return total_list, mean_list, var_list



def get_inter_hidden_list(config_dict):
    remove_idx = set(config_dict.get("remove_layer_idx", []))
    result = []

    for layer_idx in range(12):  # assuming 12 layers: 0–11
        if str(layer_idx) not in config_dict:
            continue
        
        if layer_idx in remove_idx:
            result.append(0)
        else:
            result.append(config_dict[str(layer_idx)]["inter_hidden"])
    
    return result

class TrainingArguments:
    def __init__(
        self,
        output_dir,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        learning_rate,
        push_to_hub=False,  # TODO: add APIs for push to hub
        report_to=None,  # TODO: add APIs for wandb
        label_names=None,  # TODO: add label names
        fp16=False,  # TODO: add fp16
        weight_decay=0.01,
        dataloader_num_workers=8,
        log_interval=100,
        eval_steps=1000,  # TODO: add eval steps.
        early_stopping_patience=-1,  # TODO: add early stopping
        reorder = None,
        reorder_method = None,
        reorder_dataloader = None
    ):
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.push_to_hub = push_to_hub
        self.report_to = report_to
        self.label_names = label_names
        self.fp16 = fp16
        self.weight_decay = weight_decay
        self.dataloader_num_workers = dataloader_num_workers
        self.log_interval = log_interval
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.reorder = reorder
        self.reorder_method = reorder_method
        self.reorder_dataloader = reorder_dataloader


class Trainer:
    def __init__(
        self,
        supernet: OFM,
        args: TrainingArguments,
        data_collator,
        compute_metrics,
        train_dataset,
        eval_dataset=None,
        test_dataset=None,
        tokenizer=None,
        optimizers=None,
    ):
        self.supernet = supernet
        self.activate_model = None
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.scheduler = optimizers
        self.logger = Logger(log_dir=os.path.join(args.output_dir, "logs"))
        self.train_dataloader = self.get_train_dataloader()
        if self.eval_dataset:
            self.eval_dataloader = self.get_eval_dataloader()
        if self.test_dataset:
            self.test_dataloader = self.get_test_dataloader()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # training manager
        self.best_metric = {}

    def log_metrics(self, metrics, step, log_interval, prefix):
        self.logger.log_metrics(metrics, step, prefix=prefix)
        self.logger.print_metrics(metrics, prefix=prefix)
        if step + 1 % log_interval == 0:
            metrics = self.evaluate(self.eval_dataloader)
            self.logger.log_metrics(metrics, step, prefix=prefix)
            self.logger.print_metrics(metrics, prefix=prefix)

    def update_best_metric(self, metrics):
        if self.best_metric == {}:
            self.best_metric = metrics
            # self.supernet.save_ckpt(os.path.join(self.args.output_dir, "best_model"))
        else:
            for key in metrics:
                if key == "params":
                    continue
                if metrics[key] > self.best_metric[key]:
                    self.best_metric[key] = metrics[key]
                    self.supernet.save_ckpt(
                        os.path.join(self.args.output_dir, key + "_best_model")
                    )

    def get_train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self):

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
        )

    def get_test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def create_optimizer_and_scheduler(self):
        # TODO: if my optimizer and schedular passing by argument, skip this step
        self.optimizer = AdamW(
            self.activate_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        if self.scheduler is None:
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda=lambda x: max(0.1, 0.975**x)
            )
        else:
            self.scheduler.optimizer = self.optimizer
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda x: 0.975**x)

    def compute_loss(self, outputs, labels, soft_labels=None):
        """returns the loss"""

        if soft_labels is not None:
            kd_loss = F.kl_div(
                F.log_softmax(outputs.logits, dim=1),
                F.softmax(soft_labels.to(self.device), dim=1),
                reduction="batchmean",
            )
            return outputs.loss + kd_loss
        return outputs.loss

    def _compute_metrics(self, eval_preds):
        if self.compute_metrics is None:
            return {}
        return self.compute_metrics(eval_preds)

    def evaluate(self, eval_dataloader):
        self.activate_model.to(self.device)
        self.activate_model.eval()
        all_preds = []
        all_labels = []
        eval_preds = {}
        for batch in eval_dataloader:
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.activate_model(**batch)
                # print(outputs.predictions)
                # eval_preds = self.activate_model(**batch)
                preds = outputs.logits.detach().cpu()
                all_preds.append(preds)
                all_labels.append(batch["labels"].detach().cpu())
        batch.clear()
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        eval_preds = {"predictions": all_preds, "label_ids": all_labels}
        metrics = self._compute_metrics(eval_preds)
        metrics["params"] = self.activate_model.config.num_parameters
        return metrics

    def training_step(self, batch, soft_labels=None):
        local_grad = {k: v.cpu() for k, v in self.activate_model.state_dict().items()}

        self.activate_model.to(self.device)
        self.activate_model = nn.DataParallel(self.activate_model)

        self.activate_model.train()
        self.optimizer.zero_grad()
        outputs = self.activate_model(**batch)

        loss = self.compute_loss(
            outputs,
            labels=batch["labels"] if hasattr(batch, "labels") else None,
            soft_labels=soft_labels,
        )
        # loss.backward()
        loss.sum().backward()
        self.optimizer.step()
        self.scheduler.step()

        self.activate_model = self.activate_model.module

        with torch.no_grad():
            for k, v in self.activate_model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()

        self.supernet.apply_grad(local_grad)

        train_metrics = {
            "train_loss": loss.sum().item(),
            "params": self.activate_model.config.num_parameters,
        }
        return train_metrics

    def train(self):
        hash_table = set()
        duplicate_models = 0
        scores = vit_wanda_graph_ricci_compute(model=self.supernet.model,dataloader= self.args.reorder_dataloader, sparse_threshold=0.01)


        for i in range(100000):


            print("model sampling iteration:", i)




           

            # Train random subnets
            (
                self.activate_model,
                self.activate_model.config.num_parameters,
                self.activate_model.config.arch,
            ) = self.supernet.random_resource_aware_model()

            atten, inter, residual = extract_hidden_lists(self.activate_model.config.arch)
            key = tuple(atten + inter + residual)

            if key in hash_table:
                duplicate_models += 1
                print(f"Duplicate model found! Total duplicates so far: {duplicate_models}")
                continue
            hash_table.add(key)
            ricci_sum, ricci_mean, ricci_var = sum_scores(scores, inter)

            #metrics = self.evaluate(self.eval_dataloader)
            #print(f"Random Subnet Eval Metrics: {metrics}")

            # create a csv file and write the tuple(atten + inter + residual), metrics , params, and the ricci scores to the csv file
            with open("random_subnets2.csv", "a") as f:
                f.write(
                    #f"{key}, {metrics['metric']:.4f}, {metrics['f1']:.4f}, {ricci_sum}, {ricci_mean}, {ricci_var}, {self.activate_model.config.num_parameters}\n"
                    f"{key}, {ricci_sum}, {ricci_mean}, {ricci_var}, {self.activate_model.config.num_parameters}\n"
                    #f"{key}, {self.activate_model.config.num_parameters}\n"
                )





        return None

    