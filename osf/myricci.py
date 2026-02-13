"""
A class to compute the Ollivier-Ricci curvature of a given NetworkX graph.
"""

# Author:
#     Chien-Chun Ni
#     http://www3.cs.stonybrook.edu/~chni/

# Reference:
#     Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., & Saucan, E. 2015.
#         "Ricci curvature of the Internet topology" (Vol. 26, pp. 2758-2766).
#         Presented at the 2015 IEEE Conference on Computer Communications (INFOCOM), IEEE.
#     Ni, C.-C., Lin, Y.-Y., Gao, J., and Gu, X. 2018.
#         "Network Alignment by Discrete Ollivier-Ricci Flow", Graph Drawing 2018.
#     Ni, C.-C., Lin, Y.-Y., Luo, F. and Gao, J. 2019.
#         "Community Detection on Networks with Ricci Flow", Scientific Reports.
#     Ollivier, Y. 2009.
#         "Ricci curvature of Markov chains on metric spaces". Journal of Functional Analysis, 256(3), 810-864.


import heapq
import math
import multiprocessing as mp

from functools import lru_cache
from importlib import util

import networkit as nk
import networkx as nx
import numpy as np
import ot
import torch
import time

import torch.multiprocessing as torch_mp
torch_mp.set_start_method('spawn', force=True)

from GraphRicciCurvature.util import logger, set_verbose, cut_graph_by_cutoff, get_rf_metric_cutoff

EPSILON = 1e-7  # to prevent divided by zero

# ---Shared global variables for multiprocessing used.---
_Gk = nk.graph.Graph()
_alpha = 0.5
_weight = "weight"
_method = "OTDSinkhornMix"
_base = math.e
_exp_power = 2
_proc = mp.cpu_count()
_cache_maxsize = 1000000
_shortest_path = "all_pairs"
_nbr_topk = 3000
_OTDSinkhorn_threshold = 2000
_apsp = {}


# -------------------------------------------------------

@lru_cache(_cache_maxsize)
def _get_single_node_neighbors_distributions(node, direction="successors"):
    """Get the neighbor density distribution of given node `node`.

    Parameters
    ----------
    node : int
        Node index in Networkit graph `_Gk`.
    direction : {"predecessors", "successors"}
        Direction of neighbors in directed graph. (Default value: "successors")

    Returns
    -------
    distributions : lists of float
        Density distributions of neighbors up to top `_nbr_topk` nodes.
    nbrs : lists of int
        Neighbor index up to top `_nbr_topk` nodes.

    """
    if _Gk.isDirected():
        if direction == "predecessors":
            neighbors = list(_Gk.iterInNeighbors(node))
        else:  # successors
            neighbors = list(_Gk.iterNeighbors(node))
    else:
        neighbors = list(_Gk.iterNeighbors(node))

    # Get sum of distributions from x's all neighbors
    heap_weight_node_pair = []
    for nbr in neighbors:
        if direction == "predecessors":
            w = _base ** (-_Gk.weight(nbr, node) ** _exp_power)
        else:  # successors
            w = _base ** (-_Gk.weight(node, nbr) ** _exp_power)

        if len(heap_weight_node_pair) < _nbr_topk:
            heapq.heappush(heap_weight_node_pair, (w, nbr))
        else:
            heapq.heappushpop(heap_weight_node_pair, (w, nbr))

    nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])

    if not neighbors:
        # No neighbor, all mass stay at node
        return [1], [node]

    if nbr_edge_weight_sum > EPSILON:
        # Sum need to be not too small to prevent divided by zero
        distributions = [(1.0 - _alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]
    else:
        # Sum too small, just evenly distribute to every neighbors
        logger.warning("Neighbor weight sum too small, list:", heap_weight_node_pair)
        distributions = [(1.0 - _alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)

    nbr = [x[1] for x in heap_weight_node_pair]
    return distributions + [_alpha], nbr + [node]


def _distribute_densities(source, target):
    """Get the density distributions of source and target node, and the cost (all pair shortest paths) between
    all source's and target's neighbors. Notice that only neighbors with top `_nbr_topk` edge weights.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.
    Returns
    -------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    """

    # Distribute densities for source and source's neighbors as x
    t0 = time.time()

    if _Gk.isDirected():
        x, source_topknbr = _get_single_node_neighbors_distributions(source, "predecessors")
    else:
        x, source_topknbr = _get_single_node_neighbors_distributions(source, "successors")

    # Distribute densities for target and target's neighbors as y
    y, target_topknbr = _get_single_node_neighbors_distributions(target, "successors")

    logger.debug("%8f secs density distribution for edge." % (time.time() - t0))

    # construct the cost dictionary from x to y
    t0 = time.time()

    if _shortest_path == "pairwise":
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(_source_target_shortest_path(src, tgt))
            d.append(tmp)
        d = np.array(d)
    else:  # all_pairs
        d = _apsp[np.ix_(source_topknbr, target_topknbr)]  # transportation matrix

    x = np.array(x)     # the mass that source neighborhood initially owned
    y = np.array(y)     # the mass that target neighborhood needs to received

    logger.debug("%8f secs density matrix construction for edge." % (time.time() - t0))

    return x, y, d


@lru_cache(_cache_maxsize)
def _source_target_shortest_path(source, target):
    """Compute pairwise shortest path from `source` to `target` by BidirectionalDijkstra via Networkit.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.

    Returns
    -------
    length : float
        Pairwise shortest path length.

    """

    length = nk.distance.BidirectionalDijkstra(_Gk, source, target).run().getDistance()
    assert length < 1e300, "Shortest path between %d, %d is not found" % (source, target)
    return length


def _get_all_pairs_shortest_path():
    """Pre-compute all pairs shortest paths of the assigned graph `_Gk`."""
    logger.trace("Start to compute all pair shortest path.")

    global _Gk

    t0 = time.time()
    apsp = nk.distance.APSP(_Gk).run().getDistances()
    logger.trace("%8f secs for all pair by NetworKit." % (time.time() - t0))

    return np.array(apsp)



def _optimal_transportation_distance2(x, y, d, blur=0.01, device='cuda', n_iters=100):
    """
    Compute exact Wasserstein distance using Sinkhorn on GPU with a given cost matrix d.
    Drop-in replacement for ot.emd2.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source distribution.
    y : (n,) numpy.ndarray
        Target distribution.
    d : (m, n) numpy.ndarray
        Cost matrix.
    blur : float
        Regularization parameter (epsilon in Sinkhorn). Default 0.01.
    device : str
        'cuda' or 'cpu'.
    n_iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    m : float
        Wasserstein distance.
    """

    t0 = time.time()

    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)

    # normalize distributions
    x = x / x.sum()
    y = y / y.sum()

    # Sinkhorn algorithm
    K = torch.exp(-d / blur)  # kernel
    u = torch.ones_like(x)
    v = torch.ones_like(y)

    for _ in range(n_iters):
        u = x / (K @ v + 1e-16)
        v = y / (K.t() @ u + 1e-16)

    # transport plan
    T = torch.diag(u) @ K @ torch.diag(v)
    m = torch.sum(T * d).item()

    # logging
    # print("%8f secs for Wasserstein dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m




#GPU batched Sinkhorn cost computation from scratch
def batched_sinkhorn_cost(a, b, C, epsilon=None, n_iter=2000, tol=1e-9, device='cuda', dtype=torch.float64):
    """
    Batched version of Sinkhorn cost computation in log-domain on GPU.
    
    Parameters:
    a : (B, M) torch.Tensor - Batched source distributions (padded with 0s)
    b : (B, N) torch.Tensor - Batched target distributions (padded with 0s)
    C : (B, M, N) torch.Tensor - Batched cost matrices (padded with 0s where invalid)
    epsilon : float - Regularization (if None, auto-computed from median of finite costs)
    n_iter : int - Max iterations
    tol : float - Tolerance for convergence
    device : str
    dtype : torch.dtype
    
    Returns:
    costs : (B,) torch.Tensor - Batched Sinkhorn costs
    """
    B, M = a.shape
    _, N = b.shape
    
    # Auto epsilon if not provided
    if epsilon is None:
        # Mask to finite values (assuming no negative costs; adjust if needed)
        finite_mask = torch.isfinite(C) & (C >= 0)
        finite_C = C[finite_mask]
        scale = torch.median(finite_C) if finite_C.numel() > 0 else torch.tensor(1.0, device=device, dtype=dtype)
        epsilon = max(1e-6, float(scale) * 0.05)
    
    # Normalize (handle padded zeros naturally)
    a = a / (a.sum(dim=1, keepdim=True) + 1e-30)
    b = b / (b.sum(dim=1, keepdim=True) + 1e-30)
    
    # Precompute logK, handling inf/-inf
    logK = -C / epsilon  # Where C=0 (padded), logK=0; real costs are positive
    
    log_u = torch.zeros_like(a, dtype=dtype, device=device)
    log_v = torch.zeros_like(b, dtype=dtype, device=device)
    
    log_a = torch.log(a + 1e-16)  # log(0) -> -inf for padded
    log_b = torch.log(b + 1e-16)
    
    for i in range(n_iter):
        log_u_prev = log_u.clone()
        
        # Batched log_u update: sum over N (dim=2)
        log_u = log_a - torch.logsumexp(logK + log_v.unsqueeze(1), dim=2)
        
        # Batched log_v update: sum over M (dim=2 after transpose)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u.unsqueeze(1), dim=2)
        
        # Check convergence (max change across batch)
        if torch.max(torch.abs(log_u - log_u_prev)) < tol:
            break
    
    # Compute transport plan and cost
    logT = log_u.unsqueeze(2) + logK + log_v.unsqueeze(1)
    T = torch.exp(logT)
    
    # Mask to avoid any residual nan (though padded should be 0)
    mask = torch.isfinite(C)
    costs = torch.sum(T * C * mask, dim=(1, 2))

    # Clean up GPU memory
    del logK, log_u, log_v, logT, T
    torch.cuda.empty_cache()

    
    return costs

#GPU implementation of Sinkhorn from scratch
def sinkhorn_cost_from_cost_matrix(a, b, C, epsilon=None, n_iter=2000, tol=1e-9, device=None, dtype=torch.float64):
    """
    Stable Sinkhorn (log-domain) using exact cost matrix C on GPU/CPU.

    Parameters
    ----------
    a : (m,) numpy array
        Source mass (not necessarily normalized)
    b : (n,) numpy array
        Target mass (not necessarily normalized)
    C : (m,n) numpy array
        Cost matrix (exact distances to be used)
    epsilon : float or None
        Entropic regularization. If None, chosen adaptively from scale(C).
        Smaller epsilon -> closer to LP but requires more iterations / stability.
    n_iter : int
        Maximum Sinkhorn iterations.
    tol : float
        Stopping tolerance on change of u (in log space).
    device : None or 'cuda' / 'cpu'
        If None it will use CUDA if available.
    dtype : torch.dtype
        Use float64 for more stable numerics (slower on GPU).

    Returns
    -------
    cost : float
        Sinkhorn (regularized OT) transportation cost = sum(T * C)
    """

    # Prepare device & dtype
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Convert inputs to numpy arrays if not already
    a_np = np.asarray(a, dtype=np.float64).reshape(-1)
    b_np = np.asarray(b, dtype=np.float64).reshape(-1)
    C_np = np.asarray(C, dtype=np.float64)

    m = a_np.shape[0]
    n = b_np.shape[0]
    assert C_np.shape == (m, n), "C must be shape (len(a), len(b))"

    # Normalize masses (POT emd2 expects same total mass; here we follow that)
    suma = a_np.sum()
    sumb = b_np.sum()
    if suma == 0 or sumb == 0:
        raise ValueError("Empty distributions or zero total mass.")
    # Rescale b to have same total mass as a (POT behavior)
    b_np = b_np * (suma / (sumb + 1e-30))

    # Choose epsilon if not given:
    if epsilon is None:
        # Use scale of C. median or mean is OK. Keep epsilon a small fraction.
        scale = float(np.median(C_np)) if C_np.size > 0 else 1.0
        epsilon = max(1e-6, scale * 0.05)  # 5% of median, tweak if needed

    # Move to torch
    a_t = torch.tensor(a_np, dtype=dtype, device=dev)
    b_t = torch.tensor(b_np, dtype=dtype, device=dev)
    C_t = torch.tensor(C_np, dtype=dtype, device=dev)

    # Normalize to sum 1 for numerical stability inside iterations
    a_t = a_t / (a_t.sum() + 1e-30)
    b_t = b_t / (b_t.sum() + 1e-30)

    # Precompute logK = -C / epsilon
    logK = -C_t / float(epsilon)   # shape (m, n)

    # initialize log u and log v
    log_u = torch.zeros(m, dtype=dtype, device=dev)
    log_v = torch.zeros(n, dtype=dtype, device=dev)

    log_a = torch.log(a_t + 1e-300)
    log_b = torch.log(b_t + 1e-300)

    # Sinkhorn in log domain:
    for i in range(n_iter):
        log_u_prev = log_u

        # log_u = log(a) - logsumexp( logK + log_v[None, :] , dim=1 )
        # broadcast log_v to (1,n) then add to logK (m,n)
        log_u = log_a - torch.logsumexp(logK + log_v.unsqueeze(0), dim=1)

        # log_v = log(b) - logsumexp( logK^T + log_u[None,:] , dim=1 )
        log_v = log_b - torch.logsumexp(logK.t() + log_u.unsqueeze(0), dim=1)

        # stopping condition (max change in log_u)
        if torch.max(torch.abs(log_u - log_u_prev)) < tol:
            break

    # transport plan in log-domain: logT = log_u[:,None] + logK + log_v[None,:]
    logT = log_u.unsqueeze(1) + logK + log_v.unsqueeze(0)
    # compute T * C without creating huge intermediate in double precision
    T = torch.exp(logT)  # may be large but stable due to log-domain iter
    cost_tensor = torch.sum(T * C_t)

    cost = float(cost_tensor.cpu().item())
    return cost



def _optimal_transportation_distance(x, y, d):
    """Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    Returns
    -------
    m : float
        Optimal transportation distance.

    """

    t0 = time.time()
    m = ot.emd2(x, y, d)
    t1 = time.time()


    # just for testing purpose
    # m2 = sinkhorn_cost_from_cost_matrix(x, y, d, epsilon=0.01, n_iter=400, device='cuda')
    # t2 = time.time()
    # print(m, m2, "----", t1 - t0, t2 - t1)
    # if abs(m - m2)> 0.005:
    #     print("******")
    logger.debug(
        "%8f secs for Wasserstein dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _sinkhorn_distance(x, y, d):
    """Compute the approximate optimal transportation distance (Sinkhorn distance) of the given density distributions.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    Returns
    -------
    m : float
        Sinkhorn distance, an approximate optimal transportation distance.

    """
    t0 = time.time()
    m = ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')
    logger.debug(
        "%8f secs for Sinkhorn dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _average_transportation_distance(source, target):
    """Compute the average transportation distance (ATD) of the given density distributions.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.

    Returns
    -------
    m : float
        Average transportation distance.

    """

    t0 = time.time()
    if _Gk.isDirected():
        source_nbr = list(_Gk.iterInNeighbors(source))
    else:
        source_nbr = list(_Gk.iterNeighbors(source))
    target_nbr = list(_Gk.iterNeighbors(target))

    share = (1.0 - _alpha) / (len(source_nbr) * len(target_nbr))
    cost_nbr = 0
    cost_self = _alpha * _apsp[source][target]

    for src in source_nbr:
        for tgt in target_nbr:
            cost_nbr += _apsp[src][tgt] * share

    m = cost_nbr + cost_self  # Average transportation cost

    logger.debug("%8f secs for avg trans. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0,
                                                                                       len(source_nbr),
                                                                                       len(target_nbr)))
    return m



def _compute_ricci_curvature_single_edge(source, target):
    """Ricci curvature computation for a given single edge.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.

    Returns
    -------
    result : dict[(int,int), float]
        The Ricci curvature of given edge in dict format. E.g.: {(node1, node2): ricciCurvature}

    """
    # logger.debug("EDGE:%s,%s"%(source,target))
    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if _Gk.weight(source, target) < EPSILON:
        logger.trace("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." %
                       (source, target))
        return {(source, target): 0}

    # compute transportation distance
    m = 1  # assign an initial cost
    assert _method in ["OTD", "ATD", "Sinkhorn", "OTDSinkhornMix"], \
        'Method %s not found, support method:["OTD", "ATD", "Sinkhorn", "OTDSinkhornMix]' % _method
    if _method == "OTD":
        x, y, d = _distribute_densities(source, target)
        m = _optimal_transportation_distance(x, y, d)
    elif _method == "ATD":
        m = _average_transportation_distance(source, target)
    elif _method == "Sinkhorn":
        x, y, d = _distribute_densities(source, target)
        m = _sinkhorn_distance(x, y, d)
    elif _method == "OTDSinkhornMix":
        x, y, d = _distribute_densities(source, target)
        # When x and y are small (usually around 2000 to 3000), ot.emd2 is way faster than ot.sinkhorn2
        # So we only do sinkhorn when both x and y are too large for ot.emd2
        if len(x) > _OTDSinkhorn_threshold and len(y) > _OTDSinkhorn_threshold:
            m = _sinkhorn_distance(x, y, d)
        else:
            m = _optimal_transportation_distance(x, y, d)
            m2 = _optimal_transportation_distance2(x, y, d)

    # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
    result = 1 - (m / _Gk.weight(source, target))  # Divided by the length of d(i, j)
    logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))

    return {(source, target): result}


def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return _compute_ricci_curvature_single_edge(*stuff)

# this is the original one before refactoring
# remove the word "original" when needed
def original_compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTDSinkhornMix",
                                   base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   shortest_path="all_pairs", nbr_topk=3000):
    """Compute Ricci curvature for edges in  given edge lists.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    edge_list : list of edges
        The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])
    alpha : float
        The parameter for the discrete Ricci curvature, range from 0 ~ 1.
        It means the share of mass to leave on the original node.
        E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
        (Default value = 0.5)
    method : {"OTD", "ATD", "Sinkhorn"}
        The optimal transportation distance computation method. (Default value = "OTDSinkhornMix")

        Transportation method:
            - "OTD" for Optimal Transportation Distance,
            - "ATD" for Average Transportation Distance.
            - "Sinkhorn" for OTD approximated Sinkhorn distance.
            - "OTDSinkhornMix" use OTD for nodes of edge with less than _OTDSinkhorn_threshold(default 2000) neighbors,
            use Sinkhorn for faster computation with nodes of edge more neighbors. (OTD is faster for smaller cases)
    base : float
        Base variable for weight distribution. (Default value = `math.e`)
    exp_power : float
        Exponential power for weight distribution. (Default value = 0)
    proc : int
        Number of processor used for multiprocessing. (Default value = `cpu_count()`)
    chunksize : int
        Chunk size for multiprocessing, set None for auto decide. (Default value = `None`)
    cache_maxsize : int
        Max size for LRU cache for pairwise shortest path computation.
        Set this to `None` for unlimited cache. (Default value = 1000000)
    shortest_path : {"all_pairs","pairwise"}
        Method to compute shortest path. (Default value = `all_pairs`)
    nbr_topk : int
        Only take the top k edge weight neighbors for density distribution.
        Smaller k run faster but the result is less accurate. (Default value = 3000)

    Returns
    -------
    output : dict[(int,int), float]
        A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.

    """

    logger.trace("Number of nodes: %d" % G.number_of_nodes())
    logger.trace("Number of edges: %d" % G.number_of_edges())

    if not nx.get_edge_attributes(G, weight):
        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _cache_maxsize
    global _shortest_path
    global _nbr_topk
    global _apsp
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc
    _cache_maxsize = cache_maxsize
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if _shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        # if not _apsp:
        _apsp = _get_all_pairs_shortest_path()

    if edge_list:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
    else:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]

    # Start compute edge Ricci curvature
    t0 = time.time()

    result = []
    for arg in args:
        result.append(_wrap_compute_single_edge(arg))

    # with mp.get_context('fork').Pool(processes=_proc) as pool:
    #     # WARNING: Now only fork works, spawn will hang.

    #     # Decide chunksize following method in map_async
    #     if chunksize is None:
    #         chunksize, extra = divmod(len(args), proc * 4)
    #         if extra:
    #             chunksize += 1

    #     # Compute Ricci curvature for edges
    #     result = pool.imap_unordered(_wrap_compute_single_edge, args, chunksize=chunksize)
    #     pool.close()
    #     pool.join()

    # Convert edge index from nk back to nx for final output
    output = {}
    for rc in result:
        for k in list(rc.keys()):
            output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]

    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return output



def _compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTDSinkhornMix",
                                   base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   shortest_path="all_pairs", nbr_topk=3000, batch_size=2048, num_gpus=1):
    """Compute Ricci curvature for edges in given edge lists (with streaming batches and optional multi-GPU support)."""

    logger.trace("Number of nodes: %d" % G.number_of_nodes())
    logger.trace("Number of edges: %d" % G.number_of_edges())

    if not nx.get_edge_attributes(G, weight):
        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _cache_maxsize
    global _shortest_path
    global _nbr_topk
    global _apsp
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc  # Ignored in batched version
    _cache_maxsize = cache_maxsize
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    apsp = None
    if shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        apsp = _get_all_pairs_shortest_path()
        _apsp = apsp
    try:
        if edge_list:
            edge_list_nk = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
        else:
            edge_list_nk = [(nx2nk_ndict[source], nk2nx_ndict[target]) for source, target in G.edges()]
    except KeyError as e:
        logger.error("Edge list contains nodes not in graph: %s" % str(e))
        raise
    # Start compute edge Ricci curvature
    t0 = time.time()

    output = {}

    if method not in ["Sinkhorn", "OTDSinkhornMix"]:
        # Fallback to sequential for other methods (e.g., ATD, OTD)
        print("Using sequential computation for method:", method)
        for source, target in edge_list_nk:
            if _Gk.weight(source, target) < EPSILON:
                output[(nk2nx_ndict[source], nk2nx_ndict[target])] = 0
                continue
            rc = _wrap_compute_single_edge((source, target))
            for k in list(rc.keys()):
                output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]
    else:
        # Detect available GPUs
        available_gpus = torch.cuda.device_count()
        if num_gpus < 0:
            num_gpus = available_gpus
        num_gpus = min(num_gpus, available_gpus)
        if num_gpus == 0:
            device = 'cpu'
            logger.warning("No GPUs detected, falling back to CPU.")
        else:
            logger.info(f"Using {num_gpus} GPUs for computation.")

        if num_gpus <= 1:
            # Single GPU/CPU: Use streaming batches as before
            device = 'cuda:0' if num_gpus > 0 else 'cpu'
            num_edges = len(edge_list_nk)
            for start in range(0, num_edges, batch_size):
                end = min(start + batch_size, num_edges)
                sub_x, sub_y, sub_d, sub_edges_nk = [], [], [], []
                for i in range(start, end):
                    source, target = edge_list_nk[i]
                    if _Gk.weight(source, target) < EPSILON:
                        output[(nk2nx_ndict[source], nk2nx_ndict[target])] = 0
                        continue
                    x, y, d = _distribute_densities(source, target)
                    sub_x.append(x)
                    sub_y.append(y)
                    sub_d.append(d)
                    sub_edges_nk.append((source, target))

                sub_B = len(sub_x)
                if sub_B == 0:
                    continue

                max_m = max(len(xi) for xi in sub_x)
                max_n = max(len(yi) for yi in sub_y)

                batched_a = torch.zeros((sub_B, max_m), device=device, dtype=torch.float64)
                batched_b = torch.zeros((sub_B, max_n), device=device, dtype=torch.float64)
                batched_C = torch.zeros((sub_B, max_m, max_n), device=device, dtype=torch.float64)

                for i in range(sub_B):
                    m = len(sub_x[i])
                    n = len(sub_y[i])
                    batched_a[i, :m] = torch.tensor(sub_x[i], device=device, dtype=torch.float64)
                    batched_b[i, :n] = torch.tensor(sub_y[i], device=device, dtype=torch.float64)
                    batched_C[i, :m, :n] = torch.tensor(sub_d[i], device=device, dtype=torch.float64)

                costs = batched_sinkhorn_cost(batched_a, batched_b, batched_C, epsilon=None, n_iter=100, tol=1e-9, device=device)

                for i in range(sub_B):
                    source, target = sub_edges_nk[i]
                    m = costs[i].item()
                    ricci = 1 - (m / _Gk.weight(source, target))
                    output[(nk2nx_ndict[source], nk2nx_ndict[target])] = ricci

                logger.debug(f"Processed batch {start // batch_size + 1} ({end}/{num_edges} edges)")
        else:
            # Multi-GPU: Split edges across GPUs using torch.multiprocessing
            # Split edge_list_nk into num_gpus chunks
            chunk_size = (len(edge_list_nk) + num_gpus - 1) // num_gpus
            chunks = [edge_list_nk[i:i + chunk_size] for i in range(0, len(edge_list_nk), chunk_size)]

            # Define worker function for each GPU
            def worker(gpu_id, chunk, shared_output, G, weight, alpha, method, base, exp_power, shortest_path, nbr_topk, cache_maxsize, nk2nx_ndict, apsp):
                global _Gk
                global _alpha
                global _weight
                global _method
                global _base
                global _exp_power
                global _cache_maxsize
                global _shortest_path
                global _nbr_topk
                global _apsp

                _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
                _alpha = alpha
                _weight = weight
                _method = method
                _base = base
                _exp_power = exp_power
                _cache_maxsize = cache_maxsize
                _shortest_path = shortest_path
                _nbr_topk = nbr_topk
                if shortest_path == "all_pairs":
                    _apsp = apsp

                device = f'cuda:{gpu_id}'
                local_output = {}
                num_chunk_edges = len(chunk)
                for start in range(0, num_chunk_edges, batch_size):
                    end = min(start + batch_size, num_chunk_edges)
                    sub_x, sub_y, sub_d, sub_edges_nk = [], [], [], []
                    for source, target in chunk[start:end]:
                        if _Gk.weight(source, target) < EPSILON:
                            local_output[(nk2nx_ndict[source], nk2nx_ndict[target])] = 0
                            continue
                        x, y, d = _distribute_densities(source, target)
                        sub_x.append(x)
                        sub_y.append(y)
                        sub_d.append(d)
                        sub_edges_nk.append((source, target))

                    sub_B = len(sub_x)
                    if sub_B == 0:
                        continue

                    max_m = max(len(xi) for xi in sub_x)
                    max_n = max(len(yi) for yi in sub_y)

                    batched_a = torch.zeros((sub_B, max_m), device=device, dtype=torch.float64)
                    batched_b = torch.zeros((sub_B, max_n), device=device, dtype=torch.float64)
                    batched_C = torch.zeros((sub_B, max_m, max_n), device=device, dtype=torch.float64)

                    for i in range(sub_B):
                        m = len(sub_x[i])
                        n = len(sub_y[i])
                        batched_a[i, :m] = torch.tensor(sub_x[i], device=device, dtype=torch.float64)
                        batched_b[i, :n] = torch.tensor(sub_y[i], device=device, dtype=torch.float64)
                        batched_C[i, :m, :n] = torch.tensor(sub_d[i], device=device, dtype=torch.float64)

                    costs = batched_sinkhorn_cost(batched_a, batched_b, batched_C, epsilon=None, n_iter=500, tol=1e-9, device=device)

                    for i in range(sub_B):
                        source, target = sub_edges_nk[i]
                        m = costs[i].item()
                        ricci = 1 - (m / _Gk.weight(source, target))
                        local_output[(nk2nx_ndict[source], nk2nx_ndict[target])] = ricci

                    logger.debug(f"GPU {gpu_id}: Processed batch {start // batch_size + 1} ({end}/{num_chunk_edges} edges)")

                # Add to shared output (manager dict)
                shared_output.update(local_output)

            # Use torch.multiprocessing with 'spawn' to avoid CUDA fork issues
            ctx = torch_mp.get_context('fork')
            manager = ctx.Manager()
            shared_output = manager.dict()

            processes = []
            for gpu_id in range(num_gpus):
                if gpu_id < len(chunks):
                    p = ctx.Process(target=worker, args=(gpu_id, chunks[gpu_id], shared_output, G, weight, alpha, method, base, exp_power, shortest_path, nbr_topk, cache_maxsize, nk2nx_ndict, apsp))
                    p.start()
                    processes.append(p)

            for p in processes:
                p.join()

            # Copy to main output
            output = dict(shared_output)

    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return output


# refactored version with batched processing to avoid OOM on GPU
# this works well with one GPU setup
def one_gpu_compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTDSinkhornMix",
                                   base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   shortest_path="all_pairs", nbr_topk=3000, batch_size=2048):
    """Compute Ricci curvature for edges in given edge lists (with streaming batches to avoid OOM)."""

    logger.trace("Number of nodes: %d" % G.number_of_nodes())
    logger.trace("Number of edges: %d" % G.number_of_edges())

    if not nx.get_edge_attributes(G, weight):
        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _cache_maxsize
    global _shortest_path
    global _nbr_topk
    global _apsp
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc  # Ignored in batched version
    _cache_maxsize = cache_maxsize
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if _shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        _apsp = _get_all_pairs_shortest_path()

    if edge_list:
        edge_list_nk = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
    else:
        edge_list_nk = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]

    # Start compute edge Ricci curvature
    t0 = time.time()

    output = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Fallback to CPU if no GPU

    if method in ["Sinkhorn", "OTDSinkhornMix"]:
        # Process edges in streaming batches to avoid collecting all densities/costs at once
        num_edges = len(edge_list_nk)
        for start in range(0, num_edges, batch_size):
            end = min(start + batch_size, num_edges)
            sub_x = []
            sub_y = []
            sub_d = []
            sub_edges_nk = []
            
            for i in range(start, end):
                source, target = edge_list_nk[i]
                if _Gk.weight(source, target) < EPSILON:
                    output[(nk2nx_ndict[source], nk2nx_ndict[target])] = 0
                    continue
                x, y, d = _distribute_densities(source, target)
                sub_x.append(x)
                sub_y.append(y)
                sub_d.append(d)
                sub_edges_nk.append((source, target))
            
            sub_B = len(sub_x)
            if sub_B == 0:
                continue
            
            # Pad for this sub-batch
            max_m = max(len(xi) for xi in sub_x)
            max_n = max(len(yi) for yi in sub_y)
            
            batched_a = torch.zeros((sub_B, max_m), device=device, dtype=torch.float64)
            batched_b = torch.zeros((sub_B, max_n), device=device, dtype=torch.float64)
            batched_C = torch.zeros((sub_B, max_m, max_n), device=device, dtype=torch.float64)  # Padded with 0
            
            for i in range(sub_B):
                m = len(sub_x[i])
                n = len(sub_y[i])
                batched_a[i, :m] = torch.tensor(sub_x[i], device=device, dtype=torch.float64)
                batched_b[i, :n] = torch.tensor(sub_y[i], device=device, dtype=torch.float64)
                batched_C[i, :m, :n] = torch.tensor(sub_d[i], device=device, dtype=torch.float64)
            
            # Compute batched Sinkhorn costs (force Sinkhorn even for Mix, for batching)
            costs = batched_sinkhorn_cost(batched_a, batched_b, batched_C, epsilon=None, n_iter=500, tol=1e-9, device=device)
            
            # Compute Ricci for this sub-batch
            for i in range(sub_B):
                source, target = sub_edges_nk[i]
                m = costs[i].item()
                ricci = 1 - (m / _Gk.weight(source, target))
                output[(nk2nx_ndict[source], nk2nx_ndict[target])] = ricci
            
            # Optional: Log progress for large graphs
            logger.debug(f"Processed batch {start // batch_size + 1} ({end}/{num_edges} edges)")
    
    else:
        # Fallback to sequential for other methods (e.g., ATD, OTD)
        print("Using sequential computation for method:", method)
        for source, target in edge_list_nk:
            if _Gk.weight(source, target) < EPSILON:
                output[(nk2nx_ndict[source], nk2nx_ndict[target])] = 0
                continue
            # Use the single-edge function (assuming it's defined; adjust if needed)
            rc = _wrap_compute_single_edge((source, target))
            for k in list(rc.keys()):
                output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]

    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return output

def batch_only_gpu_process_compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTDSinkhornMix",
                                   base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   shortest_path="all_pairs", nbr_topk=3000, batch_size=1024):
    """Compute Ricci curvature for edges in given edge lists (with batching for large graphs)."""

    logger.trace("Number of nodes: %d" % G.number_of_nodes())
    logger.trace("Number of edges: %d" % G.number_of_edges())

    if not nx.get_edge_attributes(G, weight):
        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _cache_maxsize
    global _shortest_path
    global _nbr_topk
    global _apsp
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc  # Ignored in batched version
    _cache_maxsize = cache_maxsize
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if _shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        _apsp = _get_all_pairs_shortest_path()

    if edge_list:
        edge_list_nk = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
    else:
        edge_list_nk = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]

    # Start compute edge Ricci curvature
    t0 = time.time()

    output = {}

    if method in ["Sinkhorn", "OTDSinkhornMix"]:
        # Collect all densities and costs
        all_x = []
        all_y = []
        all_d = []
        all_edges_nk = []
        
        for source, target in edge_list_nk:
            if _Gk.weight(source, target) < EPSILON:
                output[(nk2nx_ndict[source], nk2nx_ndict[target])] = 0
                continue
            x, y, d = _distribute_densities(source, target)
            all_x.append(x)
            all_y.append(y)
            all_d.append(d)
            all_edges_nk.append((source, target))
        
        B = len(all_x)
        if B == 0:
            return output
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Process in sub-batches to avoid OOM
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            sub_x = all_x[start:end]
            sub_y = all_y[start:end]
            sub_d = all_d[start:end]
            sub_edges_nk = all_edges_nk[start:end]
            
            sub_B = len(sub_x)
            if sub_B == 0:
                continue
            
            max_m = max(len(xi) for xi in sub_x)
            max_n = max(len(yi) for yi in sub_y)
            
            batched_a = torch.zeros((sub_B, max_m), device=device, dtype=torch.float64)
            batched_b = torch.zeros((sub_B, max_n), device=device, dtype=torch.float64)
            batched_C = torch.zeros((sub_B, max_m, max_n), device=device, dtype=torch.float64)  # Padded with 0
            
            for i in range(sub_B):
                m = len(sub_x[i])
                n = len(sub_y[i])
                batched_a[i, :m] = torch.tensor(sub_x[i], device=device, dtype=torch.float64)
                batched_b[i, :n] = torch.tensor(sub_y[i], device=device, dtype=torch.float64)
                batched_C[i, :m, :n] = torch.tensor(sub_d[i], device=device, dtype=torch.float64)
            
            # Compute batched Sinkhorn costs (force Sinkhorn even for Mix, for batching)
            costs = batched_sinkhorn_cost(batched_a, batched_b, batched_C, epsilon=None, n_iter=500, tol=1e-9, device=device)
            
            # Compute Ricci for this sub-batch and add to output
            for i in range(sub_B):
                source, target = sub_edges_nk[i]
                m = costs[i].item()
                ricci = 1 - (m / _Gk.weight(source, target))
                output[(nk2nx_ndict[source], nk2nx_ndict[target])] = ricci
    
    else:
        # Fallback to sequential for other methods (e.g., ATD, OTD)
        result = []
        for arg in edge_list_nk:
            result.append(_wrap_compute_single_edge(arg))

        # Convert edge index from nk back to nx for final output
        for rc in result:
            for k in list(rc.keys()):
                output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]

    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return output

#remove the word all_edges_togetger when needed
def all_edges_togetger_compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTDSinkhornMix",
                                   base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   shortest_path="all_pairs", nbr_topk=3000):
    """Compute Ricci curvature for edges in  given edge lists.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    edge_list : list of edges
        The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])
    alpha : float
        The parameter for the discrete Ricci curvature, range from 0 ~ 1.
        It means the share of mass to leave on the original node.
        E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
        (Default value = 0.5)
    method : {"OTD", "ATD", "Sinkhorn"}
        The optimal transportation distance computation method. (Default value = "OTDSinkhornMix")

        Transportation method:
            - "OTD" for Optimal Transportation Distance,
            - "ATD" for Average Transportation Distance.
            - "Sinkhorn" for OTD approximated Sinkhorn distance.
            - "OTDSinkhornMix" use OTD for nodes of edge with less than _OTDSinkhorn_threshold(default 2000) neighbors,
            use Sinkhorn for faster computation with nodes of edge more neighbors. (OTD is faster for smaller cases)
    base : float
        Base variable for weight distribution. (Default value = `math.e`)
    exp_power : float
        Exponential power for weight distribution. (Default value = 0)
    proc : int
        Number of processor used for multiprocessing. (Default value = `cpu_count()`)
    chunksize : int
        Chunk size for multiprocessing, set None for auto decide. (Default value = `None`)
    cache_maxsize : int
        Max size for LRU cache for pairwise shortest path computation.
        Set this to `None` for unlimited cache. (Default value = 1000000)
    shortest_path : {"all_pairs","pairwise"}
        Method to compute shortest path. (Default value = `all_pairs`)
    nbr_topk : int
        Only take the top k edge weight neighbors for density distribution.
        Smaller k run faster but the result is less accurate. (Default value = 3000)

    Returns
    -------
    output : dict[(int,int), float]
        A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.

    """

    logger.trace("Number of nodes: %d" % G.number_of_nodes())
    logger.trace("Number of edges: %d" % G.number_of_edges())

    if not nx.get_edge_attributes(G, weight):
        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _cache_maxsize
    global _shortest_path
    global _nbr_topk
    global _apsp
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc
    _cache_maxsize = cache_maxsize
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if _shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        # if not _apsp:
        _apsp = _get_all_pairs_shortest_path()



    if edge_list:
        edge_list_nk = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
    else:
        edge_list_nk = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]

    # Start compute edge Ricci curvature
    t0 = time.time()

    if method in ["Sinkhorn", "OTDSinkhornMix"]:
        # Collect all densities and costs
        all_x = []
        all_y = []
        all_d = []
        all_edges_nk = []
        
        for source, target in edge_list_nk:
            if _Gk.weight(source, target) < EPSILON:
                continue  # Skip zero-weight, but collect for consistency or handle later
            x, y, d = _distribute_densities(source, target)
            all_x.append(x)
            all_y.append(y)
            all_d.append(d)
            all_edges_nk.append((source, target))
        
        B = len(all_x)
        if B == 0:
            return {}
        
        # Pad for batching
        max_m = max(len(xi) for xi in all_x)
        max_n = max(len(yi) for yi in all_y)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Fallback to CPU if no GPU
        batched_a = torch.zeros((B, max_m), device=device, dtype=torch.float64)
        batched_b = torch.zeros((B, max_n), device=device, dtype=torch.float64)
        batched_C = torch.zeros((B, max_m, max_n), device=device, dtype=torch.float64)  # Padded with 0
        
        for i in range(B):
            m = len(all_x[i])
            n = len(all_y[i])
            batched_a[i, :m] = torch.tensor(all_x[i], device=device, dtype=torch.float64)
            batched_b[i, :n] = torch.tensor(all_y[i], device=device, dtype=torch.float64)
            batched_C[i, :m, :n] = torch.tensor(all_d[i], device=device, dtype=torch.float64)
        
        # Compute batched Sinkhorn costs (force Sinkhorn even for Mix, for batching)
        costs = batched_sinkhorn_cost(batched_a, batched_b, batched_C, epsilon=None, n_iter=500, tol=1e-9, device=device)
        
        # Compute Ricci for each
        result = {}
        for i in range(B):
            source, target = all_edges_nk[i]
            m = costs[i].item()
            ricci = 1 - (m / _Gk.weight(source, target))
            result[(nk2nx_ndict[source], nk2nx_ndict[target])] = ricci
    
    else:
        # Fallback to sequential for other methods (e.g., ATD, OTD)
        # ... (keep your existing sequential loop: result = []; for arg in edge_list_nk: result.append(_wrap_compute_single_edge(arg)))
        print("Using sequential computation for method:", method)
    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return result



def _compute_ricci_curvature(G: nx.Graph, weight="weight", **kwargs):
    """Compute Ricci curvature of edges and nodes.
    The node Ricci curvature is defined as the average of node's adjacency edges.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    **kwargs
        Additional keyword arguments passed to `_compute_ricci_curvature_edges`.

    Returns
    -------
    G: NetworkX graph
        A NetworkX graph with "ricciCurvature" on nodes and edges.
    """

    # compute Ricci curvature for all edges
    edge_ricci = _compute_ricci_curvature_edges(G, weight=weight, **kwargs)

    # Assign edge Ricci curvature from result to graph G
    nx.set_edge_attributes(G, edge_ricci, "ricciCurvature")

    # Compute node Ricci curvature
    for n in G.nodes():
        rc_sum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rc_sum += G[n][nbr]['ricciCurvature']

            # Assign the node Ricci curvature to be the average of node's adjacency edges
            G.nodes[n]['ricciCurvature'] = rc_sum / G.degree(n)
            logger.debug("node %s, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    return G


def _compute_ricci_flow(G: nx.Graph, weight="weight",
                        iterations=20, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100),
                        **kwargs
                        ):
    """Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    iterations : int
        Iterations to require Ricci flow metric. (Default value = 20)
    step : float
        step size for gradient decent process. (Default value = 1)
    delta : float
        process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)
    surgery : (function, int)
        A tuple of user define surgery function that will execute every certain iterations.
        (Default value = (lambda G, *args, **kwargs: G, 100))
    **kwargs
        Additional keyword arguments passed to `_compute_ricci_curvature`.

    Returns
    -------
    G: NetworkX graph
        A NetworkX graph with ``weight`` as Ricci flow metric.
    """

    if not nx.is_connected(G):
        logger.info("Not connected graph detected, compute on the largest connected component instead.")
        G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))

    # Set normalized weight to be the number of edges.
    normalized_weight = float(G.number_of_edges())

    global _apsp

    # Start compute edge Ricci flow
    t0 = time.time()

    if nx.get_edge_attributes(G, "original_RC"):
        logger.info("original_RC detected, continue to refine the ricci flow.")
    else:
        logger.info("No ricciCurvature detected, compute original_RC...")
        _compute_ricci_curvature(G, weight=weight, **kwargs)

        for (v1, v2) in G.edges():
            G[v1][v2]["original_RC"] = G[v1][v2]["ricciCurvature"]

        # clear the APSP since the graph have changed.
        _apsp = {}

    # Start the Ricci flow process
    for i in range(iterations):
        for (v1, v2) in G.edges():
            G[v1][v2][weight] -= step * (G[v1][v2]["ricciCurvature"]) * G[v1][v2][weight]

        # Do normalization on all weight to prevent weight expand to infinity
        w = nx.get_edge_attributes(G, weight)
        sumw = sum(w.values())
        for k, v in w.items():
            w[k] = w[k] * (normalized_weight / sumw)
        nx.set_edge_attributes(G, values=w, name=weight)
        logger.info(" === Ricci flow iteration %d === " % i)

        _compute_ricci_curvature(G, weight=weight, **kwargs)

        rc = nx.get_edge_attributes(G, "ricciCurvature")
        diff = max(rc.values()) - min(rc.values())

        logger.trace("Ricci curvature difference: %f" % diff)
        logger.trace("max:%f, min:%f | maxw:%f, minw:%f" % (
            max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

        if diff < delta:
            logger.trace("Ricci curvature converged, process terminated.")
            break

        # do surgery or any specific evaluation
        surgery_func, do_surgery = surgery
        if i != 0 and i % do_surgery == 0:
            G = surgery_func(G, weight)
            normalized_weight = float(G.number_of_edges())

        for n1, n2 in G.edges():
            logger.debug("%s %s %s" % (n1, n2, G[n1][n2]))

        # clear the APSP since the graph have changed.
        _apsp = {}

    logger.info("%8f secs for Ricci flow computation." % (time.time() - t0))

    return G


class OllivierRicci:
    """A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
    Node Ricci curvature is defined as the average of all it's adjacency edge.

    """

    def __init__(self, G: nx.Graph, weight="weight", alpha=0.5, method="OTDSinkhornMix",
                 base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, shortest_path="all_pairs",
                 cache_maxsize=1000000,
                 nbr_topk=3000, verbose="ERROR"):
        """Initialized a container to compute Ollivier-Ricci curvature/flow.

        Parameters
        ----------
        G : NetworkX graph
            A given directional or undirectional NetworkX graph.
        weight : str
            The edge weight used to compute Ricci curvature. (Default value = "weight")
        alpha : float
            The parameter for the discrete Ricci curvature, range from 0 ~ 1.
            It means the share of mass to leave on the original node.
            E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
            (Default value = 0.5)
        method : {"OTD", "ATD", "Sinkhorn"}
            The optimal transportation distance computation method. (Default value = "OTDSinkhornMix")

            Transportation method:
                - "OTD" for Optimal Transportation Distance,
                - "ATD" for Average Transportation Distance.
                - "Sinkhorn" for OTD approximated Sinkhorn distance.
                - "OTDSinkhornMix" use OTD for nodes of edge with less than _OTDSinkhorn_threshold(default 2000) neighbors,
                use Sinkhorn for faster computation with nodes of edge more neighbors. (OTD is faster for smaller cases)
        base : float
            Base variable for weight distribution. (Default value = `math.e`)
        exp_power : float
            Exponential power for weight distribution. (Default value = 2)
        proc : int
            Number of processor used for multiprocessing. (Default value = `cpu_count()`)
        chunksize : int
            Chunk size for multiprocessing, set None for auto decide. (Default value = `None`)
        shortest_path : {"all_pairs","pairwise"}
            Method to compute shortest path. (Default value = `all_pairs`)
        cache_maxsize : int
            Max size for LRU cache for pairwise shortest path computation.
            Set this to `None` for unlimited cache. (Default value = 1000000)
        nbr_topk : int
            Only take the top k edge weight neighbors for density distribution.
            Smaller k run faster but the result is less accurate. (Default value = 3000)
        verbose : {"INFO", "TRACE","DEBUG","ERROR"}
            Verbose level. (Default value = "ERROR")
                - "INFO": show only iteration process log.
                - "TRACE": show detailed iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.

        """
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.method = method
        self.base = base
        self.exp_power = exp_power
        self.proc = proc
        self.chunksize = chunksize
        self.cache_maxsize = cache_maxsize
        self.shortest_path = shortest_path
        self.nbr_topk = nbr_topk

        self.set_verbose(verbose)
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

        assert util.find_spec("ot"), \
            "Package POT: Python Optimal Transport is required for Sinkhorn distance."

        if not nx.get_edge_attributes(self.G, weight):
            logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][weight] = 1.0

        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            logger.info('Self-loop edge detected. Removing %d self-loop edges.' % len(self_loop_edges))
            self.G.remove_edges_from(self_loop_edges)

    def set_verbose(self, verbose):
        """Set the verbose level for this process.

        Parameters
        ----------
        verbose : {"INFO", "TRACE","DEBUG","ERROR"}
            Verbose level. (Default value = "ERROR")
                - "INFO": show only iteration process log.
                - "TRACE": show detailed iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.

        """
        set_verbose(verbose)

    def compute_ricci_curvature_edges(self, edge_list=None):
        """Compute Ricci curvature for edges in given edge lists.

        Parameters
        ----------
        edge_list : list of edges
            The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])

        Returns
        -------
        output : dict[(int,int), float]
            A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.
        """
        return _compute_ricci_curvature_edges(G=self.G, weight=self.weight, edge_list=edge_list,
                                              alpha=self.alpha, method=self.method,
                                              base=self.base, exp_power=self.exp_power,
                                              proc=self.proc, chunksize=self.chunksize,
                                              cache_maxsize=self.cache_maxsize, shortest_path=self.shortest_path,
                                              nbr_topk=self.nbr_topk)

    def compute_ricci_curvature(self):
        """Compute Ricci curvature of edges and nodes.
        The node Ricci curvature is defined as the average of node's adjacency edges.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "ricciCurvature" on nodes and edges.

        Examples
        --------
        To compute the Ollivier-Ricci curvature for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
            >>> orc.compute_ricci_curvature()
            >>> orc.G[0][1]
            {'weight': 1.0, 'ricciCurvature': 0.11111111071683011}
        """

        self.G = _compute_ricci_curvature(G=self.G, weight=self.weight,
                                          alpha=self.alpha, method=self.method,
                                          base=self.base, exp_power=self.exp_power,
                                          proc=self.proc, chunksize=self.chunksize, cache_maxsize=self.cache_maxsize,
                                          shortest_path=self.shortest_path,
                                          nbr_topk=self.nbr_topk)
        return self.G

    def compute_ricci_flow(self, iterations=10, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
        """Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

        Parameters
        ----------
        iterations : int
            Iterations to require Ricci flow metric. (Default value = 10)
        step : float
            Step size for gradient decent process. (Default value = 1)
        delta : float
            Process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)
        surgery : (function, int)
            A tuple of user define surgery function that will execute every certain iterations.
            (Default value = (lambda G, *args, **kwargs: G, 100))

        Returns
        -------
        G: NetworkX graph
            A graph with ``weight`` as Ricci flow metric.

        Examples
        --------
        To compute the Ollivier-Ricci flow for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
            >>> orc_OTD.compute_ricci_flow(iterations=10)
            >>> orc_OTD.G[0][1]
            {'weight': 0.06399135316908759,
             'ricciCurvature': 0.18608249978652802,
             'original_RC': 0.11111111071683011}
        """
        self.G = _compute_ricci_flow(G=self.G, weight=self.weight,
                                     iterations=iterations, step=step, delta=delta, surgery=surgery,
                                     alpha=self.alpha, method=self.method,
                                     base=self.base, exp_power=self.exp_power,
                                     proc=self.proc, chunksize=self.chunksize, cache_maxsize=self.cache_maxsize,
                                     shortest_path=self.shortest_path, nbr_topk=self.nbr_topk)
        return self.G

    def ricci_community(self, cutoff_step=0.025, drop_threshold=0.01):
        """Detect community clustering by Ricci flow metric.
        The communities are detected by the modularity drop while iteratively remove edge weight (Ricci flow metric)
        from large to small.

        Parameters
        ----------
        cutoff_step: float
            The step size to find the good cutoff points.
        drop_threshold: float
            At least drop this much to considered as a drop for good_cut.

        Returns
        -------
        cutoff: float
            Ricci flow metric weight cutoff for detected community clustering.
        clustering : dict
            Detected community clustering.

        Examples
        --------
        To compute the Ricci community for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
            >>> orc.compute_ricci_flow(iterations=50)
            >>> cc = orc.ricci_community()
            >>> print("The detected community label of node 0: %s" % cc[1][0])
            The detected community label of node 0: 0
        """

        cc = self.ricci_community_all_possible_clusterings(cutoff_step=cutoff_step, drop_threshold=drop_threshold)
        assert cc, "No clustering found!"

        number_of_clustering = len(set(cc[-1][1].values()))
        logger.info("Communities detected: %d" % number_of_clustering)

        return cc[-1]

    def ricci_community_all_possible_clusterings(self, cutoff_step=0.025, drop_threshold=0.01):
        """Detect community clustering by Ricci flow metric (all possible clustering guesses).
        The communities are detected by Modularity drop while iteratively remove edge weight (Ricci flow metric)
        from large to small.

        Parameters
        ----------
        cutoff_step: float
            The step size to find the good cutoff points.
        drop_threshold: float
            At least drop this much to considered as a drop for good_cut.

        Returns
        -------
        cc : list of (float, dict)
            All detected cutoff and community clusterings pairs. Clusterings are detected by detected cutoff points from
            large to small. Usually the last one is the best clustering result.

        Examples
        --------
        To compute the Ricci community for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
            >>> orc.compute_ricci_flow(iterations=50)
            >>> cc = orc.ricci_community_all_possible_clusterings()
            >>> print("The number of possible clusterings: %d" % len(cc))
            The number of possible clusterings: 3
        """

        if not nx.get_edge_attributes(self.G, "original_RC"):
            logger.info("Ricci flow not detected yet, run Ricci flow with default setting first...")
            self.compute_ricci_flow()

        logger.info("Ricci flow detected, start cutting graph into community...")
        cut_guesses = \
            get_rf_metric_cutoff(self.G, weight=self.weight, cutoff_step=cutoff_step, drop_threshold=drop_threshold)
        assert cut_guesses, "No cutoff point found!"

        Gp = self.G.copy()
        cc = []
        for cut in cut_guesses[::-1]:
            Gp = cut_graph_by_cutoff(Gp, cutoff=cut, weight=self.weight)
            # Get connected component after cut as clustering
            cc.append((cut, {c: idx for idx, comp in enumerate(nx.connected_components(Gp)) for c in comp}))

        return cc
