"""
Utility functions for CellDiffusion package.

This module contains helper functions for data preprocessing, postprocessing,
and other utility functions.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from typing import Optional, Tuple, Union
import warnings


def preprocess_adata(
    adata,
    min_counts: int = 3,
    target_sum: float = 1e4,
    n_top_genes: int = 2304,
    log_transform: bool = True,
    highly_variable: bool = True,
    copy: bool = False
):
    """
    Preprocess AnnData object for CellDiffusion.
    
    Args:
        adata: AnnData object to preprocess
        min_counts: Minimum number of counts for gene filtering
        target_sum: Target sum for normalization
        n_top_genes: Number of highly variable genes to select
        log_transform: Whether to apply log1p transformation
        highly_variable: Whether to select highly variable genes
        copy: Whether to return a copy
        
    Returns:
        AnnData: Preprocessed AnnData object
    """
    if copy:
        adata = adata.copy()
    
    # Store raw counts
    adata.layers["counts"] = adata.X.copy()
    
    # Filter genes
    sc.pp.filter_genes(adata, min_counts=min_counts)
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log transform
    if log_transform:
        sc.pp.log1p(adata)
    
    # Store normalized data
    adata.raw = adata
    
    # Select highly variable genes
    if highly_variable:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
        )
    
    return adata


def adata_to_tensor(
    adata, 
    image_size: int = 48,
    cluster_key: Optional[str] = None,
    cluster_value: Optional[str] = None
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Convert AnnData object to tensor format suitable for diffusion model.
    
    Args:
        adata: AnnData object
        image_size: Size to reshape the gene expression data to (image_size x image_size)
        cluster_key: Key in adata.obs for cluster information
        cluster_value: Specific cluster value to filter for
        
    Returns:
        torch.Tensor: Tensor of shape (n_cells, 1, image_size, image_size)
        np.ndarray: Array of cell indices used
        np.ndarray: Gene names used (length image_size*image_size, with padding if needed)
    """
    # Get expression data
    if hasattr(adata.X, 'toarray'):
        data = adata.X.toarray()
    else:
        data = adata.X
    var_names = np.array(adata.var_names)
    
    # Filter by cluster if specified
    if cluster_key is not None and cluster_value is not None:
        cluster_mask = adata.obs[cluster_key] == cluster_value
        data = data[cluster_mask]
        cell_indices = np.where(cluster_mask)[0]
    else:
        cell_indices = np.arange(data.shape[0])
    
    # Check if data can be reshaped to square
    n_genes = data.shape[1]
    required_genes = image_size * image_size
    
    if n_genes < required_genes:
        # Pad with zeros
        padding = np.zeros((data.shape[0], required_genes - n_genes))
        data = np.concatenate([data, padding], axis=1)
        # Construct padded gene names
        pad_names = np.array([f"PAD_{i}" for i in range(required_genes - n_genes)])
        used_gene_names = np.concatenate([var_names, pad_names])
        warnings.warn(f"Data has {n_genes} genes but requires {required_genes}. Padding with zeros.")
    elif n_genes > required_genes:
        # Take top genes or truncate
        data = data[:, :required_genes]
        used_gene_names = var_names[:required_genes]
        warnings.warn(f"Data has {n_genes} genes but requires {required_genes}. Truncating to first {required_genes} genes.")
    else:
        used_gene_names = var_names
    
    # Reshape to square format
    n_cells = data.shape[0]
    data_tensor = torch.FloatTensor(data).reshape(n_cells, 1, image_size, image_size)
    
    return data_tensor, cell_indices, used_gene_names


def tensor_to_adata(
    tensor: Union[torch.Tensor, np.ndarray],
    original_adata,
    cell_indices: Optional[np.ndarray] = None,
    pseudo_cell_prefix: str = "pseudo_",
    image_size: int = 48,
    gene_names: Optional[Union[np.ndarray, list]] = None
):
    """
    Convert tensor back to AnnData format with pseudo-cell annotations.
    
    Args:
        tensor: Generated tensor data
        original_adata: Original AnnData object for reference
        cell_indices: Indices of original cells used for generation
        pseudo_cell_prefix: Prefix for pseudo-cell names
        image_size: Size of the square image representation
        
    Returns:
        AnnData: New AnnData object with pseudo-cells
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Flatten tensor back to gene expression format
    if len(tensor.shape) == 4:  # (n_cells, channels, height, width)
        n_cells = tensor.shape[0]
        data = tensor.reshape(n_cells, -1)
    else:
        data = tensor
    
    # Determine var (genes) to use
    if gene_names is None:
        # Fallback: use original adata var_names (truncate if needed)
        n_original_genes = original_adata.shape[1]
        if data.shape[1] > n_original_genes:
            data = data[:, :n_original_genes]
        var_names_out = np.array(original_adata.var_names[:data.shape[1]])
    else:
        gene_names = np.array(gene_names)
        # Ensure lengths match data dimension
        if len(gene_names) != data.shape[1]:
            if len(gene_names) > data.shape[1]:
                var_names_out = gene_names[:data.shape[1]]
            else:
                # pad gene names if somehow shorter
                pad_needed = data.shape[1] - len(gene_names)
                pad_names = np.array([f"PAD_{i}" for i in range(pad_needed)])
                var_names_out = np.concatenate([gene_names, pad_names])
        else:
            var_names_out = gene_names
    
    # Create new AnnData object
    obs_names = [f"{pseudo_cell_prefix}{i}" for i in range(data.shape[0])]
    
    # Create the AnnData object
    pseudo_adata = sc.AnnData(
        X=data,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=pd.Index(var_names_out, name=original_adata.var.index.name))
    )
    
    # Add pseudo-cell annotation
    pseudo_adata.obs['cell_type'] = 'pseudo_cell'
    pseudo_adata.obs['is_pseudo'] = True
    
    # Add reference information if cell_indices provided
    if cell_indices is not None and len(cell_indices) > 0:
        # Map to reference clusters if available
        for key in original_adata.obs.columns:
            if key in ['leiden', 'louvain'] or 'cluster' in key.lower():
                # Get the most common cluster from the reference cells
                ref_clusters = original_adata.obs.iloc[cell_indices][key]
                if not ref_clusters.empty:
                    most_common_cluster = ref_clusters.mode().iloc[0] if len(ref_clusters.mode()) > 0 else ref_clusters.iloc[0]
                    pseudo_adata.obs[f'reference_{key}'] = most_common_cluster
    
    return pseudo_adata


def combine_adata(original_adata, pseudo_adata, copy: bool = True):
    """
    Combine original AnnData with pseudo-cells.
    
    Args:
        original_adata: Original AnnData object
        pseudo_adata: AnnData object with pseudo-cells
        copy: Whether to return a copy of the original data
        
    Returns:
        AnnData: Combined AnnData object
    """
    if copy:
        original_adata = original_adata.copy()
    
    # Add pseudo-cell indicators to original data
    if 'is_pseudo' not in original_adata.obs.columns:
        original_adata.obs['is_pseudo'] = False
    
    # Ensure consistent var (genes)
    common_genes = original_adata.var_names.intersection(pseudo_adata.var_names)
    original_subset = original_adata[:, common_genes]
    pseudo_subset = pseudo_adata[:, common_genes]
    
    # Concatenate
    combined_adata = sc.concat(
        [original_subset, pseudo_subset],
        axis=0,
        join='outer',
        index_unique=None
    )
    
    return combined_adata


def postprocess_results(
    generated_tensor: Union[torch.Tensor, np.ndarray],
    original_adata,
    cell_indices: Optional[np.ndarray] = None,
    combine_with_original: bool = True,
    pseudo_cell_prefix: str = "pseudo_",
    image_size: int = 48,
    gene_names: Optional[Union[np.ndarray, list]] = None
):
    """
    Complete postprocessing pipeline for generated pseudo-cells.
    
    Args:
        generated_tensor: Generated tensor from diffusion model
        original_adata: Original AnnData object
        cell_indices: Indices of original cells used for generation
        combine_with_original: Whether to combine with original data
        pseudo_cell_prefix: Prefix for pseudo-cell names
        image_size: Size of the square image representation
        
    Returns:
        AnnData: Processed AnnData object with pseudo-cells
    """
    # Convert tensor to AnnData
    pseudo_adata = tensor_to_adata(
        generated_tensor,
        original_adata,
        cell_indices=cell_indices,
        pseudo_cell_prefix=pseudo_cell_prefix,
        image_size=image_size,
        gene_names=gene_names
    )
    
    # Combine with original if requested
    if combine_with_original:
        result_adata = combine_adata(original_adata, pseudo_adata)
    else:
        result_adata = pseudo_adata
    
    return result_adata


def calculate_gene_padding_size(n_genes: int) -> int:
    """
    Calculate the square image size needed for a given number of genes.
    
    Args:
        n_genes: Number of genes
        
    Returns:
        int: Square image size
    """
    return int(np.ceil(np.sqrt(n_genes)))


def validate_adata(adata) -> bool:
    """
    Validate AnnData object for CellDiffusion processing.
    
    Args:
        adata: AnnData object to validate
        
    Returns:
        bool: True if valid, raises ValueError if not
    """
    if not hasattr(adata, 'X'):
        raise ValueError("AnnData object must have expression data in .X")
    
    if adata.X.shape[0] == 0:
        raise ValueError("AnnData object has no cells")
    
    if adata.X.shape[1] == 0:
        raise ValueError("AnnData object has no genes")
    
    return True


def get_device():
    """
    Get the best available device for computation.
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




