import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
import copy
from collections import defaultdict
from .myricci import OllivierRicci as myOllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import torch.nn.functional as F
import numpy as np
from .my_forman_ricci import FormanRicciGPU






def mlp_masking(model, sparcity=.5, method='magnitude'):
    sam_vit_layers = model.vision_encoder.layers

    # Ensure sparsity is between 0 and 100
    assert 0 <= sparcity <= 1, "Sparcity should be a value between 0 and 1."

    # Convert the percentage to a fraction
    fraction = sparcity

    for i, layer in enumerate(sam_vit_layers):
        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias  # bias of lin1

        # Flatten weights to easily sort and prune
        W1_flat = W1.view(-1)
        W2_flat = W2.view(-1)

        # Determine the number of elements to mask
        num_to_mask_W1 = int(fraction * W1_flat.numel())
        num_to_mask_W2 = int(fraction * W2_flat.numel())

        if method == 'magnitude':
            # Sort W1 and W2 by absolute magnitude and get the threshold values
            threshold_W1 = torch.topk(W1_flat.abs(), num_to_mask_W1, largest=False).values.max()
            threshold_W2 = torch.topk(W2_flat.abs(), num_to_mask_W2, largest=False).values.max()

            # Mask out the parameters below the threshold by setting them to zero
            W1_mask = W1.abs() >= threshold_W1
            W2_mask = W2.abs() >= threshold_W2
        
        else:
            # Generate random indices for W1 and W2 to mask
            W1_indices_to_mask = torch.randperm(W1_flat.numel())[:num_to_mask_W1]
            W2_indices_to_mask = torch.randperm(W2_flat.numel())[:num_to_mask_W2]

            # Create masks initialized to all ones (keep all)
            W1_mask = torch.ones_like(W1_flat)
            W2_mask = torch.ones_like(W2_flat)

            # Set the random indices to zero (mask them)
            W1_mask[W1_indices_to_mask] = 0
            W2_mask[W2_indices_to_mask] = 0

            # Reshape the masks back to original dimensions of W1 and W2
            W1_mask = W1_mask.view(W1.shape)
            W2_mask = W2_mask.view(W2.shape)

        # Apply the mask to W1 and W2
        W1.data.mul_(W1_mask)
        W2.data.mul_(W2_mask)

        # Optionally: You could also mask the biases similarly if needed

        print(f"Layer {i}: Masked {num_to_mask_W1} params in W1 and {num_to_mask_W2} params in W2.")

    return model



wanda_sums = {i:[[],[]] for i in range(12)}

# Assuming encoder is already a deep copy of model.vision_encoder
def randomize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight)  # You can use other initializations too
            if layer.bias is not None:
                init.zeros_(layer.bias)


def mlp_forward_hook2(inst, inp, out, layer, lin):
    W = inst.weight  # shape: (3072, 768)

    #print(f'inst : {inst} \t layer : {layer} \t lin : {lin}')
    #print(f'\tW : {W.shape}')

    C_out = W.shape[1]
    l2_norm = inp[0].view(-1,C_out)
    l2_norm = l2_norm.norm(p=2, dim=0)

    #print(f'\tl2_norm : {l2_norm.shape}')

    wanda = W.abs() * l2_norm

    if lin == 1:
        #.sum(dim=1)
        row_sums = torch.abs(wanda)
        wanda_sums[layer][0].append(row_sums)
    
    elif lin == 2:
        #.sum(dim=0)
        column_sums = torch.abs(wanda)
        wanda_sums[layer][1].append(column_sums)
    
    #print(f'\twanda : {wanda.shape}')

    #return wanda

def mlp_forward_hook(inst, inp, out, layer, lin):
    W = inst.weight  # shape: (3072, 768)

    #print(f'inst : {inst} \t layer : {layer} \t lin : {lin}')
    #print(f'\tW : {W.shape}')

    C_out = W.shape[1]
    l2_norm = inp[0].view(-1,C_out)
    l2_norm = l2_norm.norm(p=2, dim=0)

    #print(f'\tl2_norm : {l2_norm.shape}')

    wanda = W.abs() * l2_norm

    if lin == 1:
        row_sums = torch.abs(wanda).sum(dim=1)
        wanda_sums[layer][0].append(row_sums)
    
    elif lin == 2:
        column_sums = torch.abs(wanda).sum(dim=0)
        wanda_sums[layer][1].append(column_sums)
    
    #print(f'\twanda : {wanda.shape}')

    #return wanda


    

def movement_reordering(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    grads = {i:[[],[]] for i in range(12)}

    loss_func = nn.MSELoss()

    encoder = copy.deepcopy(model.vision_encoder).to(device)

    # Randomize the weights of the encoder for non-zero grads
    randomize_weights(encoder)

    encoder.train()
    for idx,(inputs, labels) in enumerate(dataloader):
        data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs])}
        print(f'data["pixel_values"] : {data["pixel_values"].shape}')
        output = encoder(data["pixel_values"].to(device))

        pred_embeddings = output[0]
        gts_embeddings = torch.stack(labels).to(device)
        loss = loss_func(gts_embeddings, pred_embeddings)

        loss.backward()

        #Capture grads for lin1 and lin2
        for idx, layer in enumerate(encoder.layers):
            G1 = layer.mlp.lin1.weight.grad
            G2 = layer.mlp.lin2.weight.grad
            row_sums = G1.abs().sum(dim=1) #G1.abs()
            column_sums = G2.abs().sum(dim=0) #G2.abs()
            print(f'Layer : {idx}')
            print("\tlin1 grads:", G1.shape)
            print("\tlin2 grads:", G2.shape)
            grads[idx][0].append(row_sums)
            grads[idx][1].append(column_sums)
        
        # # Zero out gradients
        # for param in encoder.parameters():
        #     if param.grad is not None:
        #         param.grad.zero_()
    
    score_dist = {}
    print(f'Aggregating movement sums')
    for (k,v),layer in zip(grads.items(),encoder.layers):
        grad_row_sums = sum(v[0]) / len(v[0])
        grad_column_sums = sum(v[1]) / len(v[1])

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias

        weight_row_sums = W1.abs().sum(dim=1) #W1.abs() 
        weight_column_sums = W2.abs().sum(dim=0) #W2.abs()
        
        avg_row_sums = grad_row_sums.abs() * weight_row_sums
        avg_column_sums = grad_column_sums.abs() * weight_column_sums

        avg_sums = (avg_row_sums + avg_column_sums) / 2

        score_dist[k] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True)

        print(f'{k} --> {avg_sums.shape}')
        # print(f'\tgrad_row_sums : {grad_row_sums}')
        # print(f'\tweight_row_sums : {weight_row_sums}')
        # print(f'\tavg_row_sums : {avg_row_sums}')
        # print(f'\tavg_sums : {avg_sums}')

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist

def wanda_reordering(model,dataloader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = model.vision_encoder.to(device)

    hooks_1, hooks_2 = [],[]


    for idx, layer in enumerate(encoder.layers):
        hook_1 = layer.mlp.lin1.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=1)) #module.register_backward_hook)
        hook_2 = layer.mlp.lin2.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=2))
        hooks_1.append(hook_1)
        hooks_2.append(hook_2)
    
    #encoder = nn.DataParallel(encoder)
    encoder.eval()

    with torch.no_grad():
        for idx,(inputs) in enumerate(dataloader):
            #data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs])}
            #print(f'data["pixel_values"] : {data["pixel_values"].shape}')
            output = encoder(inputs["pixel_values"].to(device))
    
        for hook_1,hook_2 in zip(hooks_1,hooks_2):
            hook_1.remove()
            hook_2.remove()

    score_dist = {}
    #print(f'Aggregating wanda sums')
    for (k,v),layer in zip(wanda_sums.items(),encoder.layers):
        avg_sums = ((sum(v[0]) / len(v[0])) + (sum(v[1]) / len(v[1]))) / 2
        #print(f'{k} --> {avg_sums.shape}')

        score_dist[k] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True)

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist


def magnitude_reordering(sam_vit_layers):

    score_dist = {}
    
    for i, layer in enumerate(sam_vit_layers):

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias
        
        row_sums = W1.sum(dim=1)
        column_sums = W2.sum(dim=0)
        avg_sums = (row_sums + column_sums) / 2
        score_dist[i] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True) #descending=True

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist


def mask_layers(model, layer_indices_to_mask):
    """
    Masks specified layers in the model by setting their parameters to zero.

    Args:
        model (torch.nn.Module): The model containing the layers.
        layer_indices_to_mask (list): List of layer indices to mask.

    Returns:
        torch.nn.Module: The modified model with masked layers.
    """
    for idx, layer in enumerate(model.vision_encoder.layers):
        if idx in layer_indices_to_mask:
            # Zero out the parameters in the attention sub-layer
            layer.attn.qkv.weight.data.zero_()
            layer.attn.qkv.bias.data.zero_()
            layer.attn.proj.weight.data.zero_()
            layer.attn.proj.bias.data.zero_()

            # Zero out the parameters in the MLP sub-layer
            layer.mlp.lin1.weight.data.zero_()
            layer.mlp.lin1.bias.data.zero_()
            layer.mlp.lin2.weight.data.zero_()
            layer.mlp.lin2.bias.data.zero_()

            # Zero out the LayerNorm parameters if desired (optional)
            layer.layer_norm1.weight.data.zero_()
            layer.layer_norm1.bias.data.zero_()
            layer.layer_norm2.weight.data.zero_()
            layer.layer_norm2.bias.data.zero_()

    return model

def remove_layers(model, layer_indices_to_remove):
    """
    Removes specified layers from the model by their indices.

    Args:
        model (torch.nn.Module): The model containing the layers.
        layer_indices_to_remove (list): List of layer indices to remove.

    Returns:
        torch.nn.Module: The modified model with specified layers removed.
    """
    # Sort the indices in descending order to avoid index shifting issues
    layer_indices_to_remove = sorted(layer_indices_to_remove, reverse=True)
    
    # Iterate over the indices and remove the corresponding layers
    for idx in layer_indices_to_remove:
        del model.vision_encoder.layers[idx]
    
    return model


def sam_weight_reorder(model, dataloader=None, method='magnitude'):
    """_summary_

    Args:
        model (torch.module): Pytorch model
        order (int, optional): Order used to compute importance. Defaults to 0.

    Returns:
        torch.module: Model
    """

    if method == 'wanda':
        score_dist = wanda_reordering(model, dataloader)

    elif method == 'magnitude':
        #model = model.to('cpu')
        sam_vit_layers = model.vision_encoder.layers
        score_dist = magnitude_reordering(sam_vit_layers)
    elif method == 'movement':
        score_dist = movement_reordering(model,dataloader)
        
    return model, score_dist





# work fine (samani)
def vit_magnitude_reordering(vit_layers):
    """Compute magnitude-based importance scores for ViT MLP blocks."""
    score_dist = {}
    for i, layer in enumerate(vit_layers):
        W1 = layer.intermediate.dense.weight  # [3072, 768]
        W2 = layer.output.dense.weight  # [768, 3072]
        b1 = layer.intermediate.dense.bias
        row_sums = W1.abs().sum(dim=1)
        column_sums = W2.abs().sum(dim=0)
        avg_sums = (row_sums + column_sums) / 2
        score_dist[i] = avg_sums.flatten().tolist()
        _, sorted_indices = avg_sums.sort(descending=True)
        W1_sorted = W1[sorted_indices, :]
        W2_sorted = W2[:, sorted_indices]
        b1_sorted = b1[sorted_indices]
        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted
    return score_dist

# work fine (samani)
def vit_wanda_reordering(model, dataloader):
    """Compute Wanda-based importance scores for ViT MLP blocks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    global wanda_sums
    wanda_sums = {i: [[], []] for i in range(len(model.vit.encoder.layer))}
    hooks_1, hooks_2 = [], []
    for idx, layer in enumerate(model.vit.encoder.layer):
        hook_1 = layer.intermediate.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=1))
        hook_2 = layer.output.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=2))
        hooks_1.append(hook_1)
        hooks_2.append(hook_2)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["pixel_values"].to(device)
            model(inputs)
    for hook_1, hook_2 in zip(hooks_1, hooks_2):
        hook_1.remove()
        hook_2.remove()
    score_dist = []
    for idx, layer in enumerate(model.vit.encoder.layer):
        avg_sums = ((sum(wanda_sums[idx][0]) / len(wanda_sums[idx][0])) + (sum(wanda_sums[idx][1]) / len(wanda_sums[idx][1]))) / 2
        score_dist.append(avg_sums)
        _, sorted_indices = avg_sums.sort(descending=True)
        W1 = layer.intermediate.dense.weight
        W2 = layer.output.dense.weight
        b1 = layer.intermediate.dense.bias
        W1_sorted = W1[sorted_indices, :]
        W2_sorted = W2[:, sorted_indices]
        b1_sorted = b1[sorted_indices]
        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted
    return score_dist


    # work fine (samani)
def vit_wanda_ricci_reordering(model, dataloader):
    sparse_threshold=0.1
    """Compute Wanda-based importance scores for ViT MLP blocks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    global wanda_sums
    wanda_sums = {i: [[], []] for i in range(len(model.vit.encoder.layer))}
    hooks_1, hooks_2 = [], []
    for idx, layer in enumerate(model.vit.encoder.layer):
        hook_1 = layer.intermediate.dense.register_forward_hook(partial(mlp_forward_hook2, layer=idx, lin=1))
        hook_2 = layer.output.dense.register_forward_hook(partial(mlp_forward_hook2, layer=idx, lin=2))
        hooks_1.append(hook_1)
        hooks_2.append(hook_2)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["pixel_values"].to(device)
            model(inputs)
    for hook_1, hook_2 in zip(hooks_1, hooks_2):
        hook_1.remove()
        hook_2.remove()
    score_dist = []
    for idx, layer in enumerate(model.vit.encoder.layer):




        wanda_score = ((sum(wanda_sums[idx][0]) / len(wanda_sums[idx][0])) + (sum(wanda_sums[idx][1]) / len(wanda_sums[idx][1]))) / 2
        W1 = (sum(wanda_sums[idx][0]) / len(wanda_sums[idx][0])).detach().cpu().numpy()
        W2 = (sum(wanda_sums[idx][1]) / len(wanda_sums[idx][1])).detach().cpu().numpy()

        D_hid, D_in = W1.shape[0],layer.intermediate.dense.in_features
        
        # Build Graph
        G = nx.DiGraph()
        max_w1 = np.max(np.abs(W1))
        max_w2 = np.max(np.abs(W2))
        thresh_w1 = sparse_threshold * max_w1
        thresh_w2 = sparse_threshold * max_w2
        
        # Vectorized Edge Addition for Speed
        rows, cols = np.where(np.abs(W1) >= thresh_w1)
        for r, c in zip(rows, cols):
            # W1: Input(c) -> Hidden(r)
            # Store hidden_idx=r on edge
            G.add_edge(c, D_in + r, weight=1.0/(np.abs(W1[r,c])+1e-6), layer='W1', hidden_idx=r)
            
        rows, cols = np.where(np.abs(W2) >= thresh_w2)
        for r, c in zip(rows, cols):
            # W2: Hidden(c) -> Output(r)
            # Store hidden_idx=c on edge
            G.add_edge(D_in + c, D_in + D_hid + r, weight=1.0/(np.abs(W2[r,c])+1e-6), layer='W2', hidden_idx=c)
        
        # Compute Ricci (Fallback to zeros if empty)
        ricci_scores = np.zeros(D_hid)
        
        if G.number_of_edges() > 0:
            try:
                orc = OllivierRicci(G, alpha=alpha_orc, verbose="ERROR",proc=1)
                orc.compute_ricci_curvature()
                
                # Aggregate per neuron
                for u, v, data in orc.G.edges(data=True):
                    if 'ricciCurvature' in data:
                        # Retrieve the hidden neuron index this edge connects to/from
                        h_idx = data.get('hidden_idx')
                        if h_idx is not None:
                            ricci_scores[h_idx] += data['ricciCurvature']
            except Exception as e:
                print(f"Ricci Calc Failed for layer {i}: {e}. Using Magnitude only.")




        score_dist.append(wanda_score)
        _, sorted_indices = wanda_score.sort(descending=True)







        W1 = layer.intermediate.dense.weight
        W2 = layer.output.dense.weight
        b1 = layer.intermediate.dense.bias
        W1_sorted = W1[sorted_indices, :]
        W2_sorted = W2[:, sorted_indices]
        b1_sorted = b1[sorted_indices]
        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted
    return score_dist

def vit_movement_reordering(model, dataloader):
    """Compute movement-based importance scores for ViT MLP blocks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = copy.deepcopy(model.vit.encoder).to(device)
    encoder.train()
    loss_func = nn.CrossEntropyLoss()
    grads = {i: [[], []] for i in range(len(encoder.layer))}
    randomize_weights(encoder)
    for batch in dataloader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        encoder.zero_grad()
        outputs = encoder(inputs)[0]
        head = nn.Linear(768, 10).to(device)
        logits = head(outputs[:, 0, :])
        loss = loss_func(logits, labels)
        loss.backward()
        for idx, layer in enumerate(encoder.layer):
            G1 = layer.intermediate.dense.weight.grad
            G2 = layer.output.dense.weight.grad
            if G1 is not None and G2 is not None:
                row_sums = G1.abs().sum(dim=1)
                column_sums = G2.abs().sum(dim=0)
                grads[idx][0].append(row_sums)
                grads[idx][1].append(column_sums)
    score_dist = []
    for idx, layer in enumerate(encoder.layer):
        grad_row_sums = sum(grads[idx][0]) / len(grads[idx][0]) if grads[idx][0] else torch.zeros(3072, device=device)
        grad_column_sums = sum(grads[idx][1]) / len(grads[idx][1]) if grads[idx][1] else torch.zeros(3072, device=device)
        W1 = layer.intermediate.dense.weight
        W2 = layer.output.dense.weight
        b1 = layer.intermediate.dense.bias
        weight_row_sums = W1.abs().sum(dim=1)
        weight_column_sums = W2.abs().sum(dim=0)
        avg_row_sums = grad_row_sums.abs() * weight_row_sums
        avg_column_sums = grad_column_sums.abs() * weight_column_sums
        avg_sums = (avg_row_sums + avg_column_sums) / 2
        score_dist.append(avg_sums)
        _, sorted_indices = avg_sums.sort(descending=True)
        W1_sorted = W1[sorted_indices, :]
        W2_sorted = W2[:, sorted_indices]
        b1_sorted = b1[sorted_indices]
        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted
    return score_dist


import torch
import numpy as np
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from functools import partial
from tqdm import tqdm

# Global storage for the accumulated Wanda Matrices
# Structure: {layer_idx: {'W1': tensor, 'W2': tensor, 'count': int}}
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

def vit_wanda_graph_ricci_reordering(model, dataloader, sparse_threshold=0.01):
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
                frc = FormanRicciGPU(G,batch_size=2048, verbose="ERROR",device='cuda')
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
        
        
        

        # score_dist[i] = hybrid_scores.tolist()
        
        # --- Reorder ---
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        idx = torch.from_numpy(sorted_indices.copy()).long().to(device)
        
        # Apply to actual weights
        layer.intermediate.dense.weight.data = layer.intermediate.dense.weight.data[idx, :]
        layer.output.dense.weight.data = layer.output.dense.weight.data[:, idx]
        layer.intermediate.dense.bias.data = layer.intermediate.dense.bias.data[idx]
        
        # sort hybrid scores and store in score_dist
        score_dist[i] = torch.from_numpy(hybrid_scores[sorted_indices].copy()).tolist()

    return score_dist



def _grad_hook(module, grad_input, grad_output, storage, layer_idx):
    # grad_output[0]: [batch, seq_len, hidden_dim]
    grad = grad_output[0]
    # average over batch & seq_len → [hidden_dim]
    abs_mean = grad.abs().mean(dim=(0, 1))
    storage[layer_idx].append(abs_mean.cpu())

def vit_gradient_reordering(model, dataloader, loss_fn= nn.CrossEntropyLoss(), device=None):
    """
    Gradient‑based ViT‑MLP reordering.
    - Attach backward hooks to intermediate.dense
    - Run a forward+backward pass to collect abs‑gradients
    - Compute mean per neuron, sort descending, permute W1, b1, W2 accordingly
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    n_layers = len(model.vit.encoder.layer)
    grad_store = {i: [] for i in range(n_layers)}
    hooks = []

    # register hooks
    for idx, layer in enumerate(model.vit.encoder.layer):
        h = layer.intermediate.dense.register_full_backward_hook(
            partial(_grad_hook, storage=grad_store, layer_idx=idx)
        )
        hooks.append(h)

    # one forward+backward pass
    for batch in dataloader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs.logits, labels)

        model.zero_grad()
        loss.backward()

    # remove hooks
    for h in hooks:
        h.remove()

    # sort & permute
    score_dist = []
    for idx, layer in enumerate(model.vit.encoder.layer):
        # stack [num_batches, H] → mean → [H]
        grads = torch.stack(grad_store[idx], dim=0)
        mean_grad = grads.mean(dim=0)
        score_dist.append(mean_grad.clone())

        # descending abs‑gradient
        _, sorted_idx = mean_grad.abs().sort(descending=True)

        # permute intermediate weights & bias
        W1 = layer.intermediate.dense.weight.data    # [H, D]
        b1 = layer.intermediate.dense.bias.data      # [H]
        layer.intermediate.dense.weight.data = W1[sorted_idx, :]
        layer.intermediate.dense.bias.data   = b1[sorted_idx]

        # permute output weights (and bias, if desired)
        W2 = layer.output.dense.weight.data      # [D, H]
        b2 = layer.output.dense.bias.data        # [D] (optional)
        layer.output.dense.weight.data = W2[:, sorted_idx]
        # if you want to permute the output bias as well, you can—but usually
        # MLP output dim==hidden so b2 is shape [H] and can be permuted similarly:
        # layer.output.dense.bias.data = b2[sorted_idx]

    return score_dist


def old_vit_ricci_reordering(model, dataloader):
    """Compute Ricci-based importance scores for ViT MLP blocks using concepts from the paper 'Analyzing Neural Network Robustness Using Graph Curvature'."""
    # Note: Requires installation of GraphRicciCurvature library: pip install GraphRicciCurvature

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    num_layers = len(model.vit.encoder.layer)
    embed_dim = model.config.hidden_size
    mlp_dim = model.config.intermediate_size

    # Initialize accumulators for each layer
    accumulators = {}
    for idx in range(num_layers):
        accumulators[idx] = {
            'sum_input_abs': torch.zeros(embed_dim, device=device),
            'sum_pre_act_abs': torch.zeros(mlp_dim, device=device),
            'sum_post_act_abs': torch.zeros(mlp_dim, device=device),
            'total_tokens': 0
        }

    hooks = []
    for idx, layer in enumerate(model.vit.encoder.layer):
        def make_hook(idx, lin):
            def hook(module, inp, out):
                input_tensor = inp[0]  # (bs, seq_len, dim)
                bs, seq_len, current_dim = input_tensor.shape
                tokens = bs * seq_len
                accumulators[idx]['total_tokens'] += tokens
                abs_input = input_tensor.abs().view(tokens, current_dim)
                if lin == 1:
                    accumulators[idx]['sum_input_abs'] += abs_input.sum(dim=0)
                    abs_out = out.abs().view(tokens, out.size(-1))
                    accumulators[idx]['sum_pre_act_abs'] += abs_out.sum(dim=0)
                elif lin == 2:
                    accumulators[idx]['sum_post_act_abs'] += abs_input.sum(dim=0)  # input to lin2 is post-act
            return hook
        hook1 = layer.intermediate.dense.register_forward_hook(make_hook(idx, 1))
        hook2 = layer.output.dense.register_forward_hook(make_hook(idx, 2))
        hooks.append(hook1)
        hooks.append(hook2)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["pixel_values"].to(device)
            model(inputs)
            # Optionally clear cache after each batch to free memory
            torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    score_dist = []
    for idx, layer in enumerate(model.vit.encoder.layer):
        total_tokens = accumulators[idx]['total_tokens']
        if total_tokens == 0:
            continue  # Avoid division by zero, though unlikely

        avg_input = accumulators[idx]['sum_input_abs'] / total_tokens
        avg_pre_act = accumulators[idx]['sum_pre_act_abs'] / total_tokens
        avg_post_act = accumulators[idx]['sum_post_act_abs'] / total_tokens

        # Move averages to CPU to save GPU memory for graph computation
        avg_input = avg_input.cpu()
        avg_pre_act = avg_pre_act.cpu()
        avg_post_act = avg_post_act.cpu()

        W1 = layer.intermediate.dense.weight.data.cpu()  # (mlp_dim, embed_dim)
        b1 = layer.intermediate.dense.bias.data.cpu()  # (mlp_dim)
        W2 = layer.output.dense.weight.data.cpu()  # (embed_dim, mlp_dim)
        # b2 not used

        G = nx.DiGraph()

        # Define node ID ranges
        input_start = 0
        hidden_start = input_start + embed_dim
        output_start = hidden_start + mlp_dim

        # Add nodes as integers
        G.add_nodes_from(range(input_start, hidden_start))  # inputs: 0 to embed_dim-1
        G.add_nodes_from(range(hidden_start, output_start))  # hiddens: embed_dim to embed_dim+mlp_dim-1
        G.add_nodes_from(range(output_start, output_start + embed_dim))  # outputs: embed_dim+mlp_dim to end

        # Add edges from input to hidden with normalized weights
        for h in range(mlp_dim):
            if avg_pre_act[h] > 0:  # Approximate active neuron
                w = W1[h, :]  # (embed_dim)
                contribs = w * avg_input
                sum_contrib = contribs.sum()
                pos_mask = w > 0
                pos_sum = contribs[pos_mask].sum()
                if pos_sum > 0 and sum_contrib > 0:
                    hat_w = torch.zeros_like(w)
                    hat_w[pos_mask] = w[pos_mask] * sum_contrib / pos_sum
                    for i in range(embed_dim):
                        if hat_w[i] > 0:
                            graph_w = 1 / hat_w[i].item()
                            G.add_edge(i, hidden_start + h, weight=graph_w)

        # Add edges from hidden to output with normalized weights
        for o in range(embed_dim):
            w = W2[o, :]  # (mlp_dim)
            contribs = w * avg_post_act
            sum_contrib = contribs.sum()
            pos_mask = w > 0
            pos_sum = contribs[pos_mask].sum()
            if pos_sum > 0 and sum_contrib > 0:
                hat_w = torch.zeros_like(w)
                hat_w[pos_mask] = w[pos_mask] * sum_contrib / pos_sum
                for h in range(mlp_dim):
                    if hat_w[h] > 0:
                        graph_w = 1 / hat_w[h].item()
                        G.add_edge(hidden_start + h, output_start + o, weight=graph_w)

        # Scale edge weights for numerical stability in Sinkhorn by dividing by the maximum weight
        if G.number_of_edges() > 0:
            weights = np.array([data['weight'] for _, _, data in G.edges(data=True)])
            max_weight = np.max(weights)
            if max_weight > 0:
                for u, v in G.edges():
                    G[u][v]['weight'] /= max_weight

        # Compute Ricci curvatures with warning suppression
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR", nbr_topk=30,proc=1)

        orc.compute_ricci_curvature()

        # frc = FormanRicci(G)
        # frc.compute_ricci_curvature()

        # Compute scores for hidden neurons: -average curvature (higher for more negative, i.e., bottlenecks)
        scores = []
        for h in range(mlp_dim):
            node = hidden_start + h
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                scores.append(0.0)
            else:
                kappas = [orc.G[node][nbr]['ricciCurvature'] for nbr in neighbors]
                avg_kappa = sum(kappas) / len(kappas)
                scores.append(-avg_kappa)

        #plot histogram of scores
        # from matplotlib import pyplot as plt

        # plt.hist(scores, bins=50)
        # plt.title(f'Layer {idx} Ricci-based Importance Scores Histogram')
        # plt.xlabel('Importance Score')
        # plt.ylabel('Frequency')
        # plt.savefig(f'Directed_vit_layer_{idx}_ricci_scores_histogram.png')
        # plt.close()

        def histogram_sort(scores, num_bins=50):
            # 1. Compute histogram (and ensure float type)
            hist, bin_edges = torch.histogram(scores.float(), bins=num_bins)

            # 2. Assign bins properly (include right edge)
            bin_indices = torch.bucketize(scores, bin_edges, right=False)

            # torch.bucketize can give an index == num_bins for the max value
            # Clamp it so all indices are valid (0..num_bins-1)
            bin_indices = torch.clamp(bin_indices, max=num_bins - 1)

            # 3. Sort bins by frequency (high → low)
            bin_order = torch.argsort(hist, descending=True)

            # 4. Group element indices by bin
            bin_to_indices = {int(b): (bin_indices == b).nonzero(as_tuple=True)[0].tolist()
                            for b in range(num_bins)}

            # 5. Interleave bins by frequency
            sorted_indices = []
            max_len = max(len(v) for v in bin_to_indices.values())

            for i in range(max_len):
                for b in bin_order:
                    idxs = bin_to_indices[int(b)]
                    if i < len(idxs):
                        sorted_indices.append(idxs[i])

            # 6. Reorder tensor
            sorted_scores = scores[sorted_indices]

            print("Input size:", scores.numel())
            print("Output size:", sorted_scores.numel())

            return sorted_scores, sorted_indices


        


        #print min and max scores
        print(f'Layer {idx} Ricci Scores - min: {min(scores)}, max: {max(scores)}')
        scores_tensor = torch.tensor(scores)
        score_dist.append(scores_tensor)

        # Reorder based on descending scores
        #_, sorted_indices = scores_tensor.sort(descending=True)
        _, sorted_indices = histogram_sort(scores_tensor, num_bins=50)

        # Reorder weights and bias back on device
        W1_sorted = W1[sorted_indices, :].to(device)
        W2_sorted = W2[:, sorted_indices].to(device)
        b1_sorted = b1[sorted_indices].to(device)

        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted

        # Clear graph and other temporaries
        del G, orc
        torch.cuda.empty_cache()

    return score_dist




from tqdm import tqdm

def vit_ricci_reordering(vit_layers, sparse_threshold=0.1, alpha=0.5):
    """
    Computes Ricci-based importance scores for ViT MLP blocks and reorders neurons.
    
    Strategy:
    1. Build a sparse graph of the MLP block (Input -> Hidden -> Output).
    2. Compute ORC for all edges.
    3. Aggregated Score (Neuron n) = Sum(Ricci Curvature of all edges connected to n).
    4. Sort Ascending (Most Negative = Most Critical = Index 0).
    """
    score_dist = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Starting Ricci Reordering with threshold={sparse_threshold}...")

    for i, layer in enumerate(tqdm(vit_layers, desc="Reordering Layers")):
        # --- 1. Extract Weights ---
        W1_tensor = layer.intermediate.dense.weight # [Hidden, Input]
        W2_tensor = layer.output.dense.weight       # [Output, Hidden]
        b1_tensor = layer.intermediate.dense.bias   # [Hidden]
        
        W1 = W1_tensor.detach().cpu().numpy()
        W2 = W2_tensor.detach().cpu().numpy()
        
        D_hid, D_in = W1.shape
        D_out, _ = W2.shape
        
        # --- 2. Build Sparse Graph (Input->Hidden->Output) ---
        G = nx.DiGraph()
        
        # Thresholds
        max_w1 = np.max(np.abs(W1))
        max_w2 = np.max(np.abs(W2))
        thresh_w1 = sparse_threshold * max_w1
        thresh_w2 = sparse_threshold * max_w2
        
        # Helper mapping for Hidden Nodes:
        # Graph Node Index = D_in + hidden_index
        
        # Add W1 Edges (Input u -> Hidden v)
        # We only care about edges connected to Hidden units for scoring
        rows, cols = np.where(np.abs(W1) >= thresh_w1)
        for r, c in zip(rows, cols):
            # r is hidden_idx, c is input_idx
            u = c 
            v = D_in + r 
            dist = 1.0 / (np.abs(W1[r, c]) + 1e-6)
            G.add_edge(u, v, weight=dist, layer='W1', hidden_idx=r)

        # Add W2 Edges (Hidden u -> Output v)
        rows, cols = np.where(np.abs(W2) >= thresh_w2)
        for r, c in zip(rows, cols):
            # r is output_idx, c is hidden_idx
            u = D_in + c
            v = D_in + D_hid + r
            dist = 1.0 / (np.abs(W2[r, c]) + 1e-6)
            G.add_edge(u, v, weight=dist, layer='W2', hidden_idx=c)
            
        if G.number_of_edges() == 0:
            print(f"Warning: Layer {i} graph is empty. Skipping reordering.")
            score_dist[i] = np.zeros(D_hid).tolist()
            continue

        # --- 3. Compute Ricci Curvature ---
        # Using alpha=0.5 (Lin-Yau) or 1.0 depending on preference. 
        # Given previous results, alpha=0.5 is fine if graph is dense enough, 
        # but alpha=1.0 is often more robust for sparse graphs.
        orc = OllivierRicci(G, alpha=alpha, verbose="ERROR",proc=1)
        orc.compute_ricci_curvature()
        
        # --- 4. Aggregate Scores per Hidden Neuron ---
        # Initialize with 0.0. 
        # Neurons with NO edges (pruned by sparsity) will remain 0.
        neuron_scores = np.zeros(D_hid)
        
        for u, v, data in orc.G.edges(data=True):
            if 'ricciCurvature' in data:
                k = data['ricciCurvature']
                # We stored 'hidden_idx' in the edge data during construction
                h_idx = data['hidden_idx']
                neuron_scores[h_idx] += k
        
        score_dist[i] = neuron_scores.tolist()
        
        # --- 5. Sort and Permute ---
        # CRITICAL: Sort ASCENDING. 
        # Most negative (Critical/Bottleneck) -> First
        # Most positive (Redundant) -> Last
        sorted_indices = np.argsort(neuron_scores)
        
        # Convert to tensor
        idx = torch.from_numpy(sorted_indices.copy()).long().to(W1_tensor.device)
        
        # Apply permutation to the model in-place
        # W1: Permute rows (dim 0)
        layer.intermediate.dense.weight.data = W1_tensor[idx, :]
        # W2: Permute columns (dim 1)
        layer.output.dense.weight.data = W2_tensor[:, idx]
        # Bias: Permute (dim 0)
        layer.intermediate.dense.bias.data = b1_tensor[idx]
        
    return score_dist

def combine_scores_exponential(mag_scores, ricci_scores, lambda_val=1.0):
    """
    Combines Magnitude and Ricci using Exponential Decay.
    Score = Mag * exp(-lambda * Ricci)
    """
    # 1. Normalize Ricci to a standard deviation of 1 for stability
    # This prevents 'lambda' from needing constant tuning
    ricci_std = (ricci_scores - np.mean(ricci_scores)) / (np.std(ricci_scores) + 1e-6)
    
    # 2. Compute Multiplier
    # Negative Ricci -> Positive Exponent -> Multiplier > 1.0
    # Positive Ricci -> Negative Exponent -> Multiplier < 1.0
    multiplier = np.exp(-lambda_val * ricci_std)
    
    # 3. Combine
    hybrid_scores = mag_scores * multiplier
    
    return hybrid_scores


def vit_hybrid_reordering(vit_layers, sparse_threshold=0.1, alpha_orc=0.5, lambda_val=1.0):
    """
    Hybrid Reordering: Combines Magnitude and Ricci Curvature.
    
    Formula: Score = Magnitude * (1 - lambda * Ricci)
    
    Args:
        vit_layers: List of transformer layers.
        sparse_threshold: Threshold to prune graph edges for Ricci calc (0.05 - 0.1 recommended).
        alpha_orc: Alpha parameter for Ollivier Ricci (0.5 is Lin-Yau, 1.0 is standard).
        lambda_val: Strength of the Ricci influence. 
                    1.0 means a Ricci score of -1 doubles the importance.
                    0.1 means a Ricci score of -1 adds 10% importance.
    """
    score_dist = {}
    
    #print(f"Starting Hybrid Reordering (Mag + Ricci) | Threshold={sparse_threshold} | Lambda={lambda_val}...")

    for i, layer in enumerate(tqdm(vit_layers, desc="Hybrid Reordering")):
        # --- 1. Extract Weights ---
        W1_tensor = layer.intermediate.dense.weight # [Hidden, Input]
        W2_tensor = layer.output.dense.weight       # [Output, Hidden]
        b1_tensor = layer.intermediate.dense.bias   # [Hidden]
        
        device = W1_tensor.device
        
        # --- 2. Compute Magnitude Score (Base Score) ---
        # Standard L1 Norm Importance
        row_sums = W1_tensor.abs().sum(dim=1)     # Sum of input weights per hidden neuron
        col_sums = W2_tensor.abs().sum(dim=0)     # Sum of output weights per hidden neuron
        
        # [Hidden_Size] array of magnitude scores
        mag_scores = ((row_sums + col_sums) / 2.0).detach().cpu().numpy()
        
        # --- 3. Compute Ricci Score (The Modifier) ---
        W1 = W1_tensor.detach().cpu().numpy()
        W2 = W2_tensor.detach().cpu().numpy()
        D_hid, D_in = W1.shape
        
        # Build Graph
        G = nx.DiGraph()
        max_w1 = np.max(np.abs(W1))
        max_w2 = np.max(np.abs(W2))
        thresh_w1 = sparse_threshold * max_w1
        thresh_w2 = sparse_threshold * max_w2
        
        # Vectorized Edge Addition for Speed
        rows, cols = np.where(np.abs(W1) >= thresh_w1)
        for r, c in zip(rows, cols):
            # W1: Input(c) -> Hidden(r)
            # Store hidden_idx=r on edge
            G.add_edge(c, D_in + r, weight=1.0/(np.abs(W1[r,c])+1e-6), layer='W1', hidden_idx=r)
            
        rows, cols = np.where(np.abs(W2) >= thresh_w2)
        for r, c in zip(rows, cols):
            # W2: Hidden(c) -> Output(r)
            # Store hidden_idx=c on edge
            G.add_edge(D_in + c, D_in + D_hid + r, weight=1.0/(np.abs(W2[r,c])+1e-6), layer='W2', hidden_idx=c)
        
        # Compute Ricci (Fallback to zeros if empty)
        ricci_scores = np.zeros(D_hid)
        
        if G.number_of_edges() > 0:
            try:
                # orc = OllivierRicci(G, alpha=alpha_orc, verbose="ERROR",proc=1)
                # orc.compute_ricci_curvature()

                frc = FormanRicci(G)
                frc.compute_ricci_curvature()
                
                # Aggregate per neuron
                for u, v, data in frc.G.edges(data=True):
                    if 'formanCurvature' in data:
                        # Retrieve the hidden neuron index this edge connects to/from
                        h_idx = data.get('hidden_idx')
                        if h_idx is not None:
                            ricci_scores[h_idx] += data['formanCurvature']
            except Exception as e:
                print(f"Ricci Calc Failed for layer {i}: {e}. Using Magnitude only.")

        # --- 4. Combine: Hybrid Score Calculation ---
        
        # Normalize Ricci locally to avoid extreme outliers breaking the magnitude scale
        # We don't change the sign, just the scale.
        # If max abs ricci is > 1, we scale it down slightly to keep the multiplier stable

        max_ricci_mag = np.max(np.abs(ricci_scores))
        if max_ricci_mag > 0:
            ricci_norm = ricci_scores / (max_ricci_mag + 1e-6) 
            # Now ricci_norm is between -1 and 1
        else:
            ricci_norm = ricci_scores
        

        # FORMULA: Base * (1 - (Strength * Ricci))
        # If Ricci is -1 (Critical): Multiplier = 1 - (-1) = 2.0 (Double Importance)
        # If Ricci is +1 (Redundant): Multiplier = 1 - (1) = 0.0 (Zero Importance)

        # forman combine
        # make magnitude normilized between 0 and 1
        mag_min = np.min(mag_scores)
        mag_max = np.max(mag_scores)
        mag_norm = (mag_scores - mag_min) / (mag_max - mag_min + 1e-6)
        hybrid_scores = mag_norm - ricci_norm

        #hybrid_scores = mag_scores * (1 - (lambda_val * ricci_norm))
        
        #hybrid_scores = combine_scores_exponential(mag_scores, ricci_norm, lambda_val=lambda_val)


        
        score_dist[i] = hybrid_scores.tolist()

        # --- 5. Sort and Permute ---
        # Important: Hybrid Score means "Higher is Better"
        # So we sort DESCENDING (Largest to Smallest)
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        
        # Move indices to same device as weights to fix RuntimeError
        idx = torch.from_numpy(sorted_indices.copy()).long().to(device)
        
        # Permute Weights In-Place
        layer.intermediate.dense.weight.data = W1_tensor[idx, :]
        layer.output.dense.weight.data = W2_tensor[:, idx]
        layer.intermediate.dense.bias.data = b1_tensor[idx]
        
    return score_dist


import torch
import torch.nn.functional as F

def vit_forman_reordering(vit_layers, lambda_balance=0.5):
    """
    Reorders neurons based on Tensorized Forman-Ricci Curvature approximation.
    
    Concept: Identifies 'Hubs' that act as balanced bridges between input and output.
    Computation: O(1) matrix operations (No graph construction).
    """
    score_dist = {}
    
    print("Running Fast Forman-Ricci Reordering...")

    for i, layer in enumerate(vit_layers):
        with torch.no_grad():
            # [Hidden_Dim, Input_Dim]
            W1 = layer.intermediate.dense.weight 
            # [Output_Dim, Hidden_Dim]
            W2 = layer.output.dense.weight       
            b1 = layer.intermediate.dense.bias   
            
            # --- 1. Calculate Weighted Degrees (The 'Mass') ---
            # How much signal comes IN to this neuron?
            # Shape: [Hidden_Dim]
            degree_in = torch.norm(W1, p=1, dim=1) 
            
            # How much signal goes OUT from this neuron?
            # Shape: [Hidden_Dim]
            degree_out = torch.norm(W2, p=1, dim=0) 
            
            # --- 2. Calculate Forman-Style 'Bridge Score' ---
            # A 'Bridge' neuron must be significant in BOTH directions.
            # Simple Magnitude reordering essentially does (degree_in + degree_out).
            # Forman Curvature suggests we look at the product interaction (Quadrangles).
            
            # Geometric Mean of flow (Balanced Flow reward)
            flow_magnitude = torch.sqrt(degree_in * degree_out + 1e-8)
            
            # --- 3. The Novelty: Balance Penalty ---
            # We penalize neurons that are 'Unbalanced' (e.g., Huge In, Tiny Out).
            # These are dead ends, not bridges.
            
            # Calculate ratio (in/out) in Log space to handle scale diffs
            balance_ratio = torch.abs(torch.log(degree_in + 1e-6) - torch.log(degree_out + 1e-6))
            
            # Score = Magnitude / (1 + Unbalanced_Penalty)
            # We want High Flow + High Balance
            forman_score = flow_magnitude / (1 + lambda_balance * balance_ratio)
            
            score_dist[i] = forman_score.cpu().tolist()

            # --- 4. Sort and Permute (Descending) ---
            sorted_indices = torch.argsort(forman_score, descending=True)
            
            # Reorder weights
            layer.intermediate.dense.weight.data = W1[sorted_indices, :]
            layer.output.dense.weight.data = W2[:, sorted_indices]
            layer.intermediate.dense.bias.data = b1[sorted_indices]
            
    return score_dist


#samani for vit support
def vit_weight_reorder(model, dataloader=None, method='magnitude'):
    """
    Reorder weights in ViT's MLP blocks using specified method.

    Args:
        model (nn.Module): ViT model (e.g., ViTForImageClassification).
        dataloader (DataLoader, optional): DataLoader for data-dependent methods like Wanda.
        method (str): Reordering method ('magnitude', 'wanda', 'movement'). Defaults to 'magnitude'.

    Returns:
        tuple: (model, score_dist)
            - model: Reordered ViT model.
            - score_dist: List of importance scores for each MLP block's intermediate dimension.
    """
    if method == 'wanda':
        score_dist = vit_wanda_reordering(model, dataloader)
    elif method == 'magnitude':
        vit_layers = model.vit.encoder.layer
        score_dist = vit_magnitude_reordering(vit_layers)
    elif method == 'movement':
        score_dist = vit_movement_reordering(model, dataloader)
    elif method == 'gradient':
        score_dist = vit_gradient_reordering(model, dataloader)
    elif method == 'ricci':
        score_dist = vit_ricci_reordering(model.vit.encoder.layer, sparse_threshold=0.1, alpha=0.5)
    elif method == 'hybrid':
        score_dist = vit_hybrid_reordering(model.vit.encoder.layer, sparse_threshold=0.05, lambda_val=1.0)
    elif method == 'forman':
        score_dist = vit_forman_reordering(model.vit.encoder.layer, lambda_balance=0.4)
    elif method == 'wanda_ricci':
        #deit 0.01 vit 0.03
        score_dist = vit_wanda_graph_ricci_reordering(model,dataloader,sparse_threshold=0.01)
    elif method == 'gradient_ricci':
        score_dist = vit_gradient_graph_ricci_reordering(model,dataloader)
    
    
    
    else:
        raise ValueError(f"Unsupported reordering method: {method}")

    return model, score_dist




def compute_global_ffn_allocation(model, target_param,compute_score = 'magnitude',dataloader=None,removed_layers=None):
    """
    Use Mahdi's score formula to compute global FFN neuron allocation under a total parameter budget.
    Reorders weights in-place per layer and returns how many FFN units to keep per layer.
    """
    all_units = [] 
    model_hidden_dim = model.vit.encoder.layer[0].intermediate.dense.weight.shape[0]  # Assuming all layers have the same hidden dimension
    if compute_score == 'magnitude':
        for i, layer in enumerate(model.vit.encoder.layer):
            if removed_layers is not None and i in removed_layers:
                continue
            W1 = layer.intermediate.dense.weight     # [out_dim, in_dim]
            W2 = layer.output.dense.weight           # [in_dim, out_dim]
            b1 = layer.intermediate.dense.bias

            row_sums = W1.abs().sum(dim=1)           # [out_dim]
            column_sums = W2.abs().sum(dim=0)        # [out_dim]
            avg_sums = (row_sums + column_sums) / 2  # [out_dim]

            avg_sums = avg_sums.cpu()
            sorted_scores, sorted_indices = avg_sums.sort(descending=True)

            # Reorder weights in-place (as you already do)
            # W1_sorted = W1[sorted_indices, :]
            # W2_sorted = W2[:, sorted_indices]
            # b1_sorted = b1[sorted_indices]

            # layer.intermediate.dense.weight.data = W1_sorted
            # layer.output.dense.weight.data = W2_sorted
            # layer.intermediate.dense.bias.data = b1_sorted
            if i != 0: # Ensure the first layer keeps all units
                # normalize scores to be in the range [0, 1]
                #sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min()+1e-8)
                for j, score in enumerate(sorted_scores):
                    
                    all_units.append((score.item(), i, j))
        target_param = target_param - model_hidden_dim # Subtract the first layer's units (768) from the target_param
        # Sort all units across all layers globally
        all_units.sort(key=lambda x: x[0], reverse=True)

        all_units = all_units[:target_param]  # Keep only the top N units based on target_param
        keep_per_layer = defaultdict(int)
        for tup in all_units:
            second_value = tup[1]
            keep_per_layer[second_value] += 1
        keep_per_layer[0] = 768  # Ensure the first layer keeps all units
        return keep_per_layer
    elif compute_score == 'wanda':
        # Implement Wanda-based global FFN allocation here
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        global wanda_sums
        wanda_sums = {i: [[], []] for i in range(len(model.vit.encoder.layer))}
        hooks_1, hooks_2 = [], []
        for idx, layer in enumerate(model.vit.encoder.layer):
            
            hook_1 = layer.intermediate.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=1))
            hook_2 = layer.output.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=2))
            hooks_1.append(hook_1)
            hooks_2.append(hook_2)
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["pixel_values"].to(device)
                model(inputs)
        for hook_1, hook_2 in zip(hooks_1, hooks_2):
            hook_1.remove()
            hook_2.remove()
        score_dist = []
        for idx, layer in enumerate(model.vit.encoder.layer):
            if removed_layers is not None and idx in removed_layers:
                continue
            avg_sums = ((sum(wanda_sums[idx][0]) / len(wanda_sums[idx][0])) + (sum(wanda_sums[idx][1]) / len(wanda_sums[idx][1]))) / 2
            score_dist.append(avg_sums)
            sorted_scores, sorted_indices = avg_sums.sort(descending=True)
            # W1 = layer.intermediate.dense.weight
            # W2 = layer.output.dense.weight
            # b1 = layer.intermediate.dense.bias
            # W1_sorted = W1[sorted_indices, :]
            # W2_sorted = W2[:, sorted_indices]
            # b1_sorted = b1[sorted_indices]
            # layer.intermediate.dense.weight.data = W1_sorted
            # layer.output.dense.weight.data = W2_sorted
            # layer.intermediate.dense.bias.data = b1_sorted
            # Save individual unit scores and costs
            if idx != 0:  # Ensure the first layer keeps all units
                for j, score in enumerate(sorted_scores):
                    
                    all_units.append((score.item(), idx, j))
        all_units.sort(key=lambda x: x[0], reverse=True)
        target_param = target_param - model_hidden_dim  # Subtract the first layer's units (768) from the target_param
        all_units = all_units[:target_param]  # Keep only the top N units based on target_param
        keep_per_layer = defaultdict(int)
        for tup in all_units:
            second_value = tup[1]
            keep_per_layer[second_value] += 1
        keep_per_layer[0] = 768
        return keep_per_layer
    elif compute_score == 'gradient':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.train()  # Enable gradients

        # Zero out any previously accumulated gradients
        model.zero_grad()

        loss_fn = torch.nn.CrossEntropyLoss()

        for batch in dataloader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            loss = loss_fn(logits, labels)
            loss.backward()


        for i, layer in enumerate(model.vit.encoder.layer):
            if removed_layers is not None and i in removed_layers:
                continue

            W1 = layer.intermediate.dense.weight     # [3072, 768]
            W2 = layer.output.dense.weight           # [768, 3072]

            grad_W1 = layer.intermediate.dense.weight.grad
            grad_W2 = layer.output.dense.weight.grad

            # Importance score for each hidden unit (i.e., neuron in FFN)
            score_W1 = (W1 * grad_W1).abs().sum(dim=1)  # [3072]
            score_W2 = (W2 * grad_W2).abs().sum(dim=0)  # [3072]
            avg_scores = (score_W1 + score_W2) / 2      # [3072]

            avg_scores = avg_scores.detach().cpu()

            sorted_scores, sorted_indices = avg_scores.sort(descending=True)

            if i != 0:
                for j, score in enumerate(sorted_scores):
                    all_units.append((score.item(), i, j))

        # Sort and select top units globally
        target_param = target_param - model_hidden_dim  # reserve full size for layer 0
        all_units.sort(key=lambda x: x[0], reverse=True)
        all_units = all_units[:target_param]

        keep_per_layer = defaultdict(int)
        for tup in all_units:
            layer_id = tup[1]
            keep_per_layer[layer_id] += 1
        keep_per_layer[0] = 768

        # Important: zero gradients so they don't accumulate outside
        model.zero_grad()
        return keep_per_layer

    else:
        raise ValueError(f"Unknown compute_score: {compute_score}")



def global_vit_magnitude_reordering(vit_layers,NumberOfParams=4500):
    """Compute magnitude-based importance scores for ViT MLP blocks in all layers and then ."""
    return 0

    

def global_vit_wanda_reordering(model, dataloader,NumberOfParams=4500):
    return 0


def vit_global_weight_reorder(model, dataloader=None, method='magnitude',NumberOfParams=4500):
    """
    Reorder weights in ViT's MLP blocks using specified method.

    Args:
        model (nn.Module): ViT model (e.g., ViTForImageClassification).
        dataloader (DataLoader, optional): DataLoader for data-dependent methods like Wanda.
        method (str): Reordering method ('magnitude', 'wanda', 'movement'). Defaults to 'magnitude'.

    Returns:
        tuple: (model, score_dist)
            - model: Reordered ViT model.
            - score_dist: List of importance scores for each MLP block's intermediate dimension.
    """
    if method == 'wanda':
        score_dist = global_vit_wanda_reordering(model, dataloader,NumberOfParams=4500)
    elif method == 'magnitude':
        vit_layers = model.vit.encoder.layer
        score_dist = global_vit_magnitude_reordering(vit_layers,NumberOfParams=4500)
    else:
        raise ValueError(f"Unsupported reordering method: {method}")

    return model, score_dist