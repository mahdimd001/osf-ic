import numpy as np
import pandas as pd
import joblib
import torch


def get_indices_excluding_diagonal(tensor, threshold):
    """
    Returns (x, y) indices where tensor[x, y] > threshold AND x != y.
    """
    n = tensor.size(0)
    
    # 1. Identify values greater than the threshold
    val_mask = tensor > threshold
    
    # 2. Create a mask that is False on the diagonal and True everywhere else
    # We create an identity matrix (diagonal 1s), make it boolean, and invert it (~).
    not_diagonal_mask = ~torch.eye(n, device=tensor.device, dtype=torch.bool)
    
    # 3. Combine the masks (Bitwise AND)
    # We want positions that satisfy BOTH conditions.
    final_mask = val_mask & not_diagonal_mask
    
    # 4. Extract indices
    indices = final_mask.nonzero()
    
    return [tuple(idx.tolist()) for idx in indices]

class DPPSamplerGPU:
    def __init__(self, gamma=2.0, alpha=0.5, device='cuda'):
        self.gamma = gamma
        self.alpha = alpha
        self.device = device if torch.cuda.is_available() else 'cpu'

    def compute_similarity(self, architectures):
        """
        Computes the similarity matrix on GPU using efficient broadcasting.
        This is mathematically identical to the One-Hot method but avoids
        creating the massive 3D One-Hot tensor.
        """
        # Ensure data is on GPU
        # Shape: (N, L)
        X = torch.tensor(architectures, device=self.device, dtype=torch.long)
        n_items, length = X.shape

        # 1. Compute Hamming Distance (Mismatches)
        # We broadcast the comparison: (N, 1, L) vs (1, N, L)
        # This creates a boolean cube and sums mismatches instantly
        # Result: (N, N) matrix of mismatches
        mismatches = (X[:, None, :] != X[None, :, :]).sum(dim=-1).float()

        # 2. Normalize and Invert
        # dist_matrix / length
        normalized_dist = mismatches / length
        
        # Sim = exp(-gamma * dist)
        S = torch.exp(-self.gamma * normalized_dist)
        return S

    def sample_ids(self, architectures, scores, k):
        """
        Selects k indices using Fast Greedy MAP Inference (Schur Complement).
        """
        n = len(architectures)
        if k >= n: return list(range(n))

        # --- 1. Setup L Matrix ---
        raw_scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
        
        # Normalize scores (0 to 1)
        s_min, s_max = raw_scores.min(), raw_scores.max()
        scores_norm = (raw_scores - s_min) / (s_max - s_min + 1e-9)
        q = (scores_norm + 0.01) ** self.alpha

        # L_ij = q_i * S_ij * q_j
        S = self.compute_similarity(architectures)
        L = S * torch.outer(q, q) # Fast broadcasting for diagonal scaling

        # --- 2. Fast Greedy Selection ---
        selected = []
        
        # We maintain a boolean mask of available items to avoid re-selecting
        available_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        
        # The diagonal of L (Variance of each item)
        # We will update this variance as we select items
        current_variances = torch.diagonal(L).clone()

        for _ in range(k):
            # 1. Pick the item with the highest conditional variance (determinant gain)
            # Apply mask by setting selected items to -inf
            valid_variances = torch.where(available_mask, current_variances, torch.tensor(-float('inf'), device=self.device))
            
            best_idx = torch.argmax(valid_variances).item()
            selected.append(best_idx)
            
            # Update mask
            available_mask[best_idx] = False
            
            # Stop if we have enough
            if len(selected) == k:
                break

            # 2. Update Conditional Variances (Schur Complement Update)
            # We want to subtract the information 'explained' by the new selection.
            # New_Var(i) = Old_Var(i) - (Correlation(i, best)^2 / Var(best))
            
            # Extract correlation vector between the 'best_idx' and all other items
            # Shape: (N,)
            corr_vector = L[best_idx, :]
            
            # Variance of the selected item
            denom = current_variances[best_idx]
            
            # Avoid division by zero
            if denom < 1e-12:
                denom = 1e-12
                
            # Update step: "Orthogonalize" the remaining search space
            # This is the GPU equivalent of "finding unique items"
            # We update all variances in parallel
            update_term = (corr_vector ** 2) / denom
            current_variances = current_variances - update_term

            # Numerical stability: Clip negative variances to 0
            current_variances = torch.clamp(current_variances, min=0.0)

        return selected

class DPPSamplerOneHot:
    def __init__(self, gamma=2.0, alpha=0.5):
        """
        gamma: Similarity penalty. Higher = Forces more diversity.
        alpha: Quality weight. Lower = Allows lower scores if they are unique.
        """
        self.gamma = gamma
        self.alpha = alpha

    def _to_one_hot(self, architectures):
        """
        Converts integer architectures to One-Hot encoded 3D volume.
        Input: (N, L) e.g., (5, 12)
        Output: (N, L, Unique_Values)
        """
        # Find all unique channel sizes in the entire batch to build the vocabulary
        unique_values = np.unique(architectures)
        
        # Map values to indices (e.g., 0->0, 576->1, 1536->2)
        val_to_idx = {val: i for i, val in enumerate(unique_values)}
        
        n, l = architectures.shape
        vocab_size = len(unique_values)
        
        # Create zero-filled 3D tensor
        one_hot = np.zeros((n, l, vocab_size), dtype=np.uint8)
        
        # Fill ones
        for i in range(n):
            for j in range(l):
                val = architectures[i, j]
                idx = val_to_idx[val]
                one_hot[i, j, idx] = 1
                
        return one_hot

    def compute_similarity(self, architectures):
        """
        Computes similarity based on One-Hot overlap.
        """
        # Convert to One-Hot: (N, Length, Vocab)
        oh = self._to_one_hot(architectures)
        
        n_items = oh.shape[0]
        length = oh.shape[1]
        
        # Compute Hamming Distance on One-Hot Vectors
        # Two layers match only if they have the exact same channel count.
        # This effectively counts "Mismatched Layers"
        # We can do this efficiently: The distance is 0 if match, 1 if no match.
        
        # Using boolean broadcasting to compare entire architectures
        # Shape: (N, N, Length, Vocab)
        # Summing over last two dims gives total mismatches
        
        # Optimization: We don't need the full 4D tensor.
        # Dot product of normalized vectors also works, but let's stick to distance
        # for consistency with previous logic.
        
        dist_matrix = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(n_items):
                # Check where layers differ
                # np.all(..., axis=1) checks if the one-hot vector at that layer is identical
                matches = np.all(oh[i] == oh[j], axis=1) # Boolean array of size (Length,)
                mismatches = length - np.sum(matches)
                dist_matrix[i, j] = mismatches
        
        # Normalize distance (0.0 to 1.0)
        normalized_dist = dist_matrix / length
        
        # Convert to Similarity
        return np.exp(-self.gamma * normalized_dist)

    def sample_ids(self, architectures, scores, k):
        architectures = np.array(architectures)
        raw_scores = np.array(scores)
        n = len(architectures)

        if k >= n: return list(range(n))

        # Normalize Scores (0 to 1)
        s_min, s_max = raw_scores.min(), raw_scores.max()
        scores_norm = (raw_scores - s_min) / (s_max - s_min + 1e-9)
        q = (scores_norm + 0.01) ** self.alpha

        # Compute L
        S = self.compute_similarity(architectures)
        L = np.diag(q) @ S @ np.diag(q)

        # Greedy Selection
        selected = []
        for _ in range(k):
            best_item = -1
            best_log_det = -np.inf
            for i in range(n):
                if i in selected: continue
                
                subset = selected + [i]
                sub_L = L[np.ix_(subset, subset)]
                sign, logdet = np.linalg.slogdet(sub_L)
                
                if sign > 0 and logdet > best_log_det:
                    best_log_det = logdet
                    best_item = i
            
            if best_item != -1:
                selected.append(best_item)
            else:
                remaining = list(set(range(n)) - set(selected))
                if remaining: selected.append(remaining[0])

        return selected





# read csv input samples
def read_architectures_from_csv(file_path):
    df = pd.read_csv(file_path)       
    return df



min_size = 39.4
max_size = 85.62

NumBins = 5

#devide the range into NumBins
param_range = max_size - min_size
bin_size = param_range / NumBins

# generate architectures and assign to bins
bins = {i: [] for i in range(NumBins)}


df = read_architectures_from_csv("/lustre/hdd/LAS/jannesar-lab/msamani/OSF/all_subnets_with_predictions.csv")

# df columns: attention_1,attention_2,attention_3,attention_4,attention_5,attention_6,attention_7,attention_8,attention_9,attention_10,attention_11,attention_12,inter_hidden_1,inter_hidden_2,inter_hidden_3,inter_hidden_4,inter_hidden_5,inter_hidden_6,inter_hidden_7,inter_hidden_8,inter_hidden_9,inter_hidden_10,inter_hidden_11,inter_hidden_12,residual_1,residual_2,residual_3,residual_4,residual_5,residual_6,residual_7,residual_8,residual_9,residual_10,residual_11,residual_12,ricci_sum_1,ricci_sum_2,ricci_sum_3,ricci_sum_4,ricci_sum_5,ricci_sum_6,ricci_sum_7,ricci_sum_8,ricci_sum_9,ricci_sum_10,ricci_sum_11,ricci_sum_12,ricci_mean_1,ricci_mean_2,ricci_mean_3,ricci_mean_4,ricci_mean_5,ricci_mean_6,ricci_mean_7,ricci_mean_8,ricci_mean_9,ricci_mean_10,ricci_mean_11,ricci_mean_12,ricci_var_1,ricci_var_2,ricci_var_3,ricci_var_4,ricci_var_5,ricci_var_6,ricci_var_7,ricci_var_8,ricci_var_9,ricci_var_10,ricci_var_11,ricci_var_12,parameters,predicted_f1
# assign architectures to bins based on their parameter size
for index, row in df.iterrows():
    param_size = row['parameters']
    bin_index = int((param_size - min_size) / bin_size)
    bin_index = min(bin_index, NumBins - 1)  # Ensure bin_index is within range
    bins[bin_index].append((row['predicted_f1'], param_size, row.to_dict()))


# compute average and std accuracy of each bin
for bin_index in range(NumBins):
    bin_data = bins[bin_index]
    if len(bin_data) == 0:
        print(f"Bin {bin_index}: No architectures")
        continue

    accuracies = [item[0] for item in bin_data]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Bin {bin_index}: Num Architectures: {len(bin_data)}, Avg Accuracy: {avg_accuracy:.2f}, Std Dev: {std_accuracy:.4f}")



# keep archituctures which accuracy > avg_accuracy + 2*std_accuracy in each bin
# create refined_bins to store the selected architectures
refined_bins = {i: [] for i in range(NumBins)}
for bin_index in range(NumBins):
    bin_data = bins[bin_index]
    if len(bin_data) == 0:
        continue

    accuracies = [item[0] for item in bin_data]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    print(f"Bin {bin_index}: Num Architectures: {len(bin_data)}, Avg Accuracy: {avg_accuracy:.2f}, Std Dev: {std_accuracy:.2f}, Min Accuracy: {min_accuracy:.2f}, Max Accuracy: {max_accuracy:.2f}")


    for item in bin_data:
        if item[0] > avg_accuracy + 2*std_accuracy:
            refined_bins[bin_index].append(item)


# print the number of selected architectures in each refined bin
for bin_index in range(NumBins):
    bin_data = refined_bins[bin_index]
    print(f"Refined Bin {bin_index}: Num Selected Architectures: {len(bin_data)}")






save_add = ""
# # save refined architectures to csv separately
# for bin_index in range(NumBins):
#     bin_data = refined_bins[bin_index]
#     if len(bin_data) == 0:
#         continue

#     # create a dataframe to save
#     df_rows = []
#     for item in bin_data:
#         accuracy, size, arch_dict = item
#         row = [arch_dict[i] for i in range(12)]
#         row.append(accuracy)
#         row.append(size)
#         df_rows.append(row)

#     df_bin = pd.DataFrame(df_rows)
#     df_bin.to_csv(f"{save_add}/refined_bin_{bin_index}_architectures.csv", index=False)




# # --- USAGE ---

# candidates = np.array([
#     [1536,576,1536,1536,576,1024,1536,768,576,576,0,0],    # Arch 0
#     [1536,576,768,576,1024,1024,1024,576,1536,1536,1536,0],# Arch 1
#     [1536,576,576,1024,576,768,768,768,1536,1024,768,1536],# Arch 2
#     [1536,768,1536,768,1536,1024,1536,768,576,1536,1024,1536],# Arch 3
#     [1536,576,576,1024,576,768,768,768,1536,1024,768,1536] # Arch 4
# ])

# ricci_scores = [100.0, 80.0, 90.0, 65.0, 90.0]

# # Initialize
# sampler = DPPSamplerGPU(gamma=3.0, alpha=0.3,device='cuda')

# # Run
# selected = sampler.sample_ids(candidates, ricci_scores, k=3)
# print(f"Selected: {selected}")

# keep 15% of the architectures in each bin using DPP sampling
refined_bins2 = {i: [] for i in range(NumBins)}
for bin_index in range(NumBins):
    bin_data = refined_bins[bin_index]
    if len(bin_data) == 0:
        continue

    # Initialize
    sampler = DPPSamplerGPU(gamma=3.0, alpha=0.3,device='cuda')

    candidates = []
    ricci_scores = []
    for item in bin_data:
        accuracy, size, arch_dict = item

        # Convert arch_dict to a list of 12 values (inter_hidden_1 to inter_hidden_12)
        # also add residual_1 to the list
        arch_list = [arch_dict[f'inter_hidden_{i}'] for i in range(1, 13)]
        arch_list.append(arch_dict['residual_1'])

        candidates.append(arch_list)
        ricci_scores.append(accuracy)

    # Run
    similariy = sampler.compute_similarity(candidates)  # Precompute similarity matrix on GPU
    result = get_indices_excluding_diagonal(similariy, threshold=0.75)  # Example usage of the function to get indices above threshold
    print(f"Bin {bin_index}: Number of similar pairs above threshold: {len(result)}")

    # remove architectures that are too similar (similarity > 0.75) and keep only one of them
    to_remove = set()
    for (i, j) in result:
        if i in to_remove or j in to_remove:
            continue
        # Keep the one with higher accuracy
        if ricci_scores[i] >= ricci_scores[j]:
            to_remove.add(j)
        else:
            to_remove.add(i)

    # Remove the architectures that are too similar
    refined_bins2[bin_index] = [item for idx, item in enumerate(bin_data) if idx not in to_remove]


# print the number of selected architectures in each refined bin after removing similar architectures
for bin_index in range(NumBins):
    bin_data = refined_bins2[bin_index]
    print(f"Refined Bin {bin_index}: Num Selected Architectures after removing similar: {len(bin_data)}")


# combine all refined bins into one list
refined_architectures = []
for bin_index in range(NumBins):
    bin_data = refined_bins2[bin_index]
    refined_architectures.extend(bin_data)

import csv
def write_list_to_csv(filename, experiment_list):
    """
    Writes a list of (score, value, params_dict) tuples to a CSV file.
    Overwrites the file if it exists.
    """
    if not experiment_list:
        print("The list is empty. Nothing to write.")
        return

    # 1. Flatten the data into a format suitable for CSV
    rows_to_write = []
    for score, value, params in experiment_list:
        row = {'score': score, 'value': value}
        row.update(params) # Merges the params dict into the row
        rows_to_write.append(row)

    # 2. Determine column headers dynamically from the first item
    # (Assumes all items in the list have the same parameter keys)
    fieldnames = list(rows_to_write[0].keys())

    # 3. Write to file
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)
        print(f"Successfully wrote {len(experiment_list)} items to {filename}")
    except IOError as e:
        print(f"Error writing to file: {e}")


write_list_to_csv("refined_architectures_2var.csv", refined_architectures)

    # selected = sampler.sample_ids(candidates, ricci_scores, k=int(0.4 * len(bin_data)))
    # print(f"Bin {bin_index}: Selected Architectures: {len(selected)} out of {len(bin_data)}")
    

    # # save refined architectures to csv separately with the same format as before
    # for idx in selected:
    #     refined_bins2[bin_index].append(bin_data[idx])
    # # create a dataframe to save
    # df_rows = []
    # for item in refined_bins2[bin_index]:
    #     accuracy, size, arch_dict = item
    #     row = [arch_dict[i] for i in range(12)]
    #     row.append(accuracy)
    #     row.append(size)
    #     df_rows.append(row)

    # df_bin = pd.DataFrame(df_rows)
    # df_bin.to_csv(f"{save_add}/refined_bin_{bin_index}_architectures.csv", index=False)

