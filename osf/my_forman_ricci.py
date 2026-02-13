import networkx as nx
import torch
import math
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

def set_verbose(verbose):
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)

class FormanRicciGPU:
    def __init__(self, G: nx.Graph, weight="weight", method="augmented", 
                 device=None, batch_size=1024, verbose="ERROR"):
        """
        A GPU-accelerated class to compute Forman-Ricci curvature using PyTorch.
        
        Parameters
        ----------
        G : NetworkX graph
            Input graph.
        weight : str
            Edge/Node weight attribute key.
        method : {"1d", "augmented"}
            Computation method.
        device : str or torch.device
            "cuda", "cpu", or "mps". If None, detects automatically.
        batch_size : int
            Number of edges to process in parallel on the GPU.
        verbose : str
            Logging level.
        """
        self.G = G.copy()
        self.weight = weight
        self.method = method
        self.batch_size = batch_size

        # Device management
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        set_verbose(verbose)
        self._preprocess_graph()

    def _preprocess_graph(self):
        """Prepares graph data into PyTorch tensors."""
        logger.info(f"Preprocessing graph for {self.device}...")

        # 1. Handle defaults (fill missing weights with 1.0)
        if not nx.get_edge_attributes(self.G, self.weight):
            nx.set_edge_attributes(self.G, 1.0, self.weight)
        if not nx.get_node_attributes(self.G, self.weight):
            nx.set_node_attributes(self.G, 1.0, self.weight)
        
        if self.G.is_directed():
            logger.info("Converting directed graph to undirected for Forman-Ricci.")
            self.G = self.G.to_undirected()

        # 2. Create Mappings (Node -> Index)
        self.nodes = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        self.num_edges = self.G.number_of_edges()

        # 3. Build Tensors
        # Node Weights: (N,)
        nw_list = [self.G.nodes[n][self.weight] for n in self.nodes]
        self.node_weights = torch.tensor(nw_list, dtype=torch.float32, device=self.device)

        # Adjacency Matrix (Weighted): (N, N)
        # We use a dense matrix for fast neighbor intersection (v1 & v2).
        # Note: For massive graphs (>50k nodes), this may need SparseTensor implementation.
        adj = nx.to_numpy_array(self.G, nodelist=self.nodes, weight=self.weight)
        self.adj_matrix = torch.from_numpy(adj).float().to(self.device)

        # Edge List for iteration: (2, E)
        edges = list(self.G.edges())
        edge_indices = [[self.node_to_idx[u], self.node_to_idx[v]] for u, v in edges]
        self.edge_index = torch.tensor(edge_indices, device=self.device).t() # Shape (2, E)
        
        # Edge Weights Flat: (E,)
        ew_list = [self.G[u][v][self.weight] for u, v in edges]
        self.edge_weights_flat = torch.tensor(ew_list, dtype=torch.float32, device=self.device)

    def compute_ricci_curvature(self):
        """Compute Forman-Ricci curvature in batches on GPU."""
        
        # Array to store curvature results for edges (aligned with self.edge_index)
        edge_curvatures = torch.zeros(self.num_edges, device=self.device)
        
        logger.info(f"Starting {self.method} computation on {self.num_edges} edges...")

        # --- EDGE CURVATURE LOOP (BATCHED) ---
        num_batches = math.ceil(self.num_edges / self.batch_size)
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_edges)
            
            # Get batch indices
            batch_indices = torch.arange(start_idx, end_idx, device=self.device)
            src_idx = self.edge_index[0, batch_indices] # Shape (B,)
            dst_idx = self.edge_index[1, batch_indices] # Shape (B,)
            
            # Get scalar weights for this batch
            w_e = self.edge_weights_flat[batch_indices] # Edge weights
            w_v1 = self.node_weights[src_idx]           # Src node weights
            w_v2 = self.node_weights[dst_idx]           # Dst node weights

            if self.method == "1d":
                curv = self._compute_batch_1d(src_idx, dst_idx, w_e, w_v1, w_v2)
            elif self.method == "augmented":
                curv = self._compute_batch_augmented(src_idx, dst_idx, w_e, w_v1, w_v2)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            edge_curvatures[batch_indices] = curv
            
            if (i+1) % 10 == 0:
                logger.info(f"Processed batch {i+1}/{num_batches}")

        # --- WRITE BACK RESULTS TO GRAPH ---
        logger.info("Writing edge results back to NetworkX graph...")
        edge_curvatures_cpu = edge_curvatures.cpu().numpy()
        edges_list = list(self.G.edges())
        
        for i, (u, v) in enumerate(edges_list):
            self.G[u][v]["formanCurvature"] = float(edge_curvatures_cpu[i])

        # --- NODE CURVATURE CALCULATION ---
        # Can be done purely on GPU via scatter_add
        logger.info("Computing node curvatures...")
        
        # 1. Sum curvature of incident edges for every node
        node_curv_sum = torch.zeros(self.num_nodes, device=self.device)
        degrees = torch.zeros(self.num_nodes, device=self.device)
        
        # Add for source nodes
        node_curv_sum.index_add_(0, self.edge_index[0], edge_curvatures)
        degrees.index_add_(0, self.edge_index[0], torch.ones_like(edge_curvatures))
        
        # Add for target nodes (undirected, so edges contribute to both)
        node_curv_sum.index_add_(0, self.edge_index[1], edge_curvatures)
        degrees.index_add_(0, self.edge_index[1], torch.ones_like(edge_curvatures))
        
        # Avoid div by zero
        degrees[degrees == 0] = 1.0
        avg_node_curv = node_curv_sum / degrees
        
        # Write back to nodes
        avg_node_curv_cpu = avg_node_curv.cpu().numpy()
        for i, n in enumerate(self.nodes):
            self.G.nodes[n]["formanCurvature"] = float(avg_node_curv_cpu[i])

        logger.info("Forman curvature computation done.")

    def _compute_batch_1d(self, src, dst, w_e, w_v1, w_v2):
        """
        Formula: RC = w_e * ( w_v1/w_e + w_v2/w_e - (sum_v1_nbr + sum_v2_nbr) )
        Where sum_v1_nbr = sum( w_v1 / sqrt(w_e * w_{v1, neighbor}) )
        """
        # 1. Get neighbors mask for src and dst (Batch_Size, Num_Nodes)
        # Using index_select on the dense adjacency matrix
        src_rows = torch.index_select(self.adj_matrix, 0, src) 
        dst_rows = torch.index_select(self.adj_matrix, 0, dst)

        # 2. Mask out the edge (v1, v2) itself from the neighbor calculations
        # In the formula: v1_nbr.remove(v2)
        # We temporarily create a mask where the connection to the other node is 0
        src_rows.scatter_(1, dst.unsqueeze(1), 0)
        dst_rows.scatter_(1, src.unsqueeze(1), 0)

        # 3. Compute sums
        # Sqrt term for neighbors: sqrt(w_e * w_{neighbor_edge})
        # Note: src_rows contains edge weights to neighbors. 0 if no edge.
        
        # We need to broadcast w_e (Batch) to (Batch, Num_Nodes)
        w_e_expanded = w_e.unsqueeze(1)
        
        # Avoid division by zero for non-neighbors (where src_rows == 0)
        # We add epsilon to zeros, compute, then mask result back to 0
        epsilon = 1e-8
        
        # Term for v1 neighbors: w_v1 / sqrt(w_e * w_{v1,v})
        denom_v1 = torch.sqrt(w_e_expanded * src_rows)
        term_v1 = (w_v1.unsqueeze(1) / (denom_v1 + epsilon))
        term_v1 = term_v1 * (src_rows > 0).float() # Zero out non-neighbors
        ev1_sum = torch.sum(term_v1, dim=1)

        # Term for v2 neighbors
        denom_v2 = torch.sqrt(w_e_expanded * dst_rows)
        term_v2 = (w_v2.unsqueeze(1) / (denom_v2 + epsilon))
        term_v2 = term_v2 * (dst_rows > 0).float()
        ev2_sum = torch.sum(term_v2, dim=1)

        return w_e * ( (w_v1 / w_e) + (w_v2 / w_e) - (ev1_sum + ev2_sum) )

    def _compute_batch_augmented(self, src, dst, w_e, w_v1, w_v2):
        """
        Computes 2D (Augmented) Forman Ricci Curvature.
        Includes contributions from 'faces' (triangles).
        """
        # 1. Get Adjacency Rows (Batch, N)
        src_rows = torch.index_select(self.adj_matrix, 0, src) 
        dst_rows = torch.index_select(self.adj_matrix, 0, dst)

        # 2. Identify Faces (Triangles)
        # A node is a common neighbor if src_rows[k] > 0 AND dst_rows[k] > 0
        src_mask = (src_rows > 0).float()
        dst_mask = (dst_rows > 0).float()
        face_mask = src_mask * dst_mask # Element-wise mult is logical AND

        # 3. Identify "Parallel Neighbors" (Neighbors not part of the face)
        # remove v2 from v1 neighbors, remove v1 from v2 neighbors
        src_mask.scatter_(1, dst.unsqueeze(1), 0) 
        dst_mask.scatter_(1, src.unsqueeze(1), 0)
        
        # Neighbors of v1 exclusive of the face and v2
        v1_only_mask = src_mask - face_mask 
        v2_only_mask = dst_mask - face_mask

        # 4. Calculate Terms
        w_f = 1.0 # Assumption from original code
        
        # sum_ef: sum(w_e / w_f) for every face node
        # Since w_f is 1, this is just w_e * count(faces)
        num_faces = torch.sum(face_mask, dim=1)
        sum_ef = (w_e / w_f) * num_faces
        
        # sum_ve: sum(w_v1/w_e + w_v2/w_e)
        sum_ve = (w_v1 / w_e) + (w_v2 / w_e)
        
        # sum_ehef: Assume 0 for cycle=3 as per original code
        sum_ehef = 0.0

        # sum_veeh: Complex sum over unique neighbors
        w_e_expanded = w_e.unsqueeze(1)
        epsilon = 1e-8

        # V1 unique neighbors contribution
        denom_v1 = torch.sqrt(w_e_expanded * src_rows) + epsilon
        term_v1 = (w_v1.unsqueeze(1) / denom_v1) * v1_only_mask # Mask keeps only unique nbrs
        
        # V2 unique neighbors contribution
        denom_v2 = torch.sqrt(w_e_expanded * dst_rows) + epsilon
        term_v2 = (w_v2.unsqueeze(1) / denom_v2) * v2_only_mask

        sum_veeh = torch.sum(term_v1, dim=1) + torch.sum(term_v2, dim=1)

        return w_e * (sum_ef + sum_ve - torch.abs(sum_ehef - sum_veeh))