"""
Constructs a gene co-expression hypergraph structure using Topological Data Analysis (TDA) Mapper algorithm and UMAP projection.
"""
import numpy as np
import kmapper as km
from sklearn.cluster import DBSCAN
import torch
import umap

def get_global_adj(feature_matrix, resolution=30, overlap=0.5, eps=0.8):
    """
    Constructs a global hypergraph incidence matrix H from gene profiles via the Mapper algorithm.
         - feature_matrix: Cell-by-gene profile matrix
         - resolution: Resolution (n_cubes) for the open cover
         - overlap: Overlapping percentage between intervals
         - eps: DBSCAN distance threshold (correlation metric)
    """
    if isinstance(feature_matrix, torch.Tensor):
        feature_matrix = feature_matrix.cpu().detach().numpy()
        
    print(f"data shape: {feature_matrix.shape}")
    
    X_genes = feature_matrix.T
    
    X_genes = np.nan_to_num(X_genes) + 1e-6 * np.random.rand(*X_genes.shape)
    n_genes = X_genes.shape[0]

    reducer = umap.UMAP(
        n_components=2,
        metric='correlation',
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    projected_data = reducer.fit_transform(X_genes)

    mins = np.min(projected_data, axis=0) # [X_min, Y_min]
    maxs = np.max(projected_data, axis=0) # [X_max, Y_max]
    ranges = maxs - mins

    #  w = range / ((n_cubes - 1) * (1 - perc_overlap) + 1)
    bin_widths = ranges / ((resolution - 1) * (1.0 - overlap) + 1.0)

    padding = 0.5 * bin_widths 
    new_limits = [
        [mins[0] - padding[0], maxs[0] + padding[0]],  
        [mins[1] - padding[1], maxs[1] + padding[1]]   
    ]

    print(f" Mapper (Res={resolution}, Ov={overlap}, Eps={eps})...")
    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        projected_data,
        X_genes,
        cover=km.Cover(n_cubes=resolution, perc_overlap=overlap, limits=new_limits), 
        clusterer=DBSCAN(metric='correlation', eps=eps, min_samples=3)
    )
    
    clusters = graph['nodes'] 
    m_clusters = len(clusters)
    
    if m_clusters == 0:
        print("Warning: Mapper did not generate any clusters, degenerated to identity (self-loop) matrix.")
        return torch.eye(n_genes)

    print(f"Constructing hypergraph incidence matrix H (Genes={n_genes}, Hyperedges={m_clusters})...")
    
    H = np.zeros((n_genes, m_clusters), dtype=np.float32)
    
    for idx, (node_id, gene_indices) in enumerate(clusters.items()):
        H[gene_indices, idx] = 1.0

    valid_cols = H.sum(axis=0) > 0
    H = H[:, valid_cols]

    print(f" Final Shape: {H.shape}")
    
    return torch.FloatTensor(H)
