# CellDiffusion

A Python package for generating virtual cells using diffusion models from single-cell RNA sequencing data.

## Overview

CellDiffusion provides a generalized implementation of diffusion models for single-cell RNA sequencing data. It allows users to generate synthetic virtual cells from AnnData objects, providing a powerful tool for data augmentation and synthetic cell generation in computational biology.

## Key Features

- **Easy-to-use API**: Simple interface with AnnData input/output
- **Cluster-specific training**: Train separate models for different cell types/clusters
- **Flexible architecture**: Configurable UNet model with multiple parameters
- **Multiple noise schedules**: Linear, cosine, quadratic, and sigmoid schedules
- **Automatic preprocessing**: Built-in data preprocessing and postprocessing pipelines
- **GPU acceleration**: CUDA support for faster training and generation
- **Model checkpointing**: Save and resume training, load pre-trained models
- **Comprehensive documentation**: Extensive examples and documentation

## Installation

### From source (recommended)

```bash
git clone https://github.com/ShiltonZhang/CellDiffusion.git
cd CellDiffusion
pip install -e .
```

### Using pip

```bash
pip install celldiff
```

## Quick Start

```python
import scanpy as sc
from CellDiffusion import CellDiffusion

# Load your single-cell data
adata = sc.read_h5ad('your_data.h5ad')

# Initialize CellDiffusion
cd = CellDiffusion(
    image_size=48,          # Reshape to 48x48 (2304 genes)
    timesteps=500,          # Number of diffusion steps
)

# Preprocess data
preprocessed_adata = cd.preprocess(
    adata,
    n_top_genes=2304,      # Select highly variable genes
    log_transform=True
)

# Train the model
cd.fit(
    epochs=300,            # Training epochs
    batch_size=128,
    learning_rate=1e-3
)

# Generate pseudo-cells
result_adata = cd.generate(
    n_samples=1000,        # Generate 1000 pseudo-cells
    batch_size=500
)

# The result contains both original and pseudo-cells
print(f"Original cells: {(~result_adata.obs['is_pseudo']).sum()}")
print(f"Pseudo-cells: {result_adata.obs['is_pseudo'].sum()}")
```

## Advanced Usage

### Cluster-specific Generation

```python
# Perform clustering
sc.pp.neighbors(preprocessed_adata)
sc.tl.leiden(preprocessed_adata, key_added="leiden")

# Train cluster-specific models
cluster_results = cd.fit_by_clusters(
    cluster_key='leiden',
    epochs=200
)

# Generate from specific cluster
cluster_pseudo_adata = cd.generate(
    n_samples=500,
    cluster_key='leiden',
    cluster_value='0'  # Generate from cluster 0
)
```

### Custom Model Configuration

```python
cd = CellDiffusion(
    image_size=64,              # Larger image for more genes
    timesteps=1000,             # More diffusion steps
    beta_schedule='cosine',     # Different noise schedule
    dim=64,                     # Larger model dimension
    dim_mults=(1, 2, 4, 8),    # Custom architecture
    device='cuda'               # Use GPU
)
```

### Model Saving and Loading

```python
# Save trained model
cd.save_model('my_model.pth')

# Load pre-trained model
cd_new = CellDiffusion()
cd_new.load_model('my_model.pth')

# Generate with loaded model
new_pseudo_cells = cd_new.generate(n_samples=1000)
```

## API Reference

### CellDiffusion Class

The main class for training diffusion models and generating pseudo-cells.

#### Parameters

- `image_size` (int): Size to reshape gene expression data (default: 48)
- `timesteps` (int): Number of diffusion timesteps (default: 500) 
- `beta_schedule` (str): Noise schedule type ('linear', 'cosine', 'quadratic', 'sigmoid')
- `dim` (int): Base dimension for UNet model (default: 48)
- `dim_mults` (tuple): Dimension multipliers for UNet (default: (1, 2, 4, 8))
- `channels` (int): Number of channels (default: 1)
- `device` (str): Device to use ('cuda' or 'cpu', auto-detected if None)
- `random_seed` (int): Random seed for reproducibility (default: 42)

#### Methods

- `preprocess(adata, **kwargs)`: Preprocess AnnData object
- `fit(adata, **kwargs)`: Train the diffusion model
- `fit_by_clusters(adata, **kwargs)`: Train cluster-specific models
- `generate(n_samples, **kwargs)`: Generate pseudo-cells
- `save_model(path)`: Save trained model
- `load_model(path)`: Load pre-trained model

### Utility Functions

- `preprocess_adata()`: Preprocess AnnData with filtering, normalization, etc.
- `adata_to_tensor()`: Convert AnnData to tensor format
- `postprocess_results()`: Convert generated tensors back to AnnData
- `get_device()`: Auto-detect best available device
- `set_random_seed()`: Set random seed for reproducibility

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic training and generation
- Cluster-specific workflows  
- Advanced model configuration
- Visualization and analysis
- Model saving/loading

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- scanpy ≥ 1.9.0
- numpy ≥ 1.21.0
- pandas ≥ 1.3.0
- einops ≥ 0.6.0
- tqdm ≥ 4.64.0

See `requirements.txt` for complete dependency list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GNU General Public License v3.0 (AGPL-3.0) - see the LICENSE file for details.

## Citation

If you use CellDiffusion in your research, please cite our preprint.

### APA Style

Zhang, X., Mao, J., & Lê Cao, K.-A. (2025). CellDiffusion: a generative model to annotate single-cell and spatial RNA-seq using bulk references. BioRxiv.

### BibTeX

```bibtex
@article{zhang2025celldiffusion,
  author = {Zhang, Xiaochen and Mao, Jiadong and L{\^e} Cao, Kim-Anh},
  title = {{CellDiffusion: a generative model to annotate single-cell and spatial RNA-seq using bulk references}},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.1101/2025.10.27.684671},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {[https://www.biorxiv.org/content/10.1101/2025.10.27.684671v1](https://www.biorxiv.org/content/10.1101/2025.10.27.684671v1)}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/ShiltonZhang/CellDiffusion/issues)

## Acknowledgments

This package is based on diffusion model implementations and techniques from:
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)
- The scanpy ecosystem for single cell analysis
