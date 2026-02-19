import numpy as np
import networkx as nx
import torch
from .line import train_line  # <--- Import your new custom module
from .utils import set_seed

def generate_embeddings(norm_matrix, dimension=512, device="cuda"):
    set_seed(42)

    """
    Generates structural embeddings using the LINE algorithm (Second Order).
    Replaces the old 'ge' library with a pure PyTorch implementation.
    """
    print(f" -> Generating Embeddings using LINE (Dim: {dimension})...")
    
    # Convert numpy matrix to scipy sparse for efficient sampling if it isn't already
    # But train_line handles dense numpy arrays fine too.
    # Ideally, pass a scipy sparse matrix or a numpy array.
    import scipy.sparse as sp
    
    # Ensure it's in a format easy to process
    if not sp.issparse(norm_matrix):
        # Convert dense numpy to sparse coo_matrix for fast edge extraction
        adj_sparse = sp.coo_matrix(norm_matrix)
    else:
        adj_sparse = norm_matrix.tocoo()

    # Train LINE
    # You can tweak epochs based on dataset size. 
    # 50-100 is usually enough for Hi-C 1MB. 
    # For high res, increase to 500.
    embeddings = train_line(
        adj_sparse, 
        embedding_dim=dimension, 
        device=device,
        epochs=100,       # Increased epochs for stability
        batch_size=4096,  # Larger batch size for GPU efficiency
        neg_ratio=5
    )
    
    return embeddings
