"""
CellDiffusion: A Python package for generating pseudo-cells using diffusion models

This package provides a generalized implementation of diffusion models for 
single-cell RNA sequencing data, allowing users to generate pseudo-cells 
from AnnData objects.
"""

from .core import CellDiffusion
from .models import Unet
from .diffusion import DiffusionSampler
from .utils import preprocess_adata, postprocess_results

__version__ = "0.1.0"
__author__ = "Xiaochen Zhang"
__email__ = "xiaochen.zhang3@student.unimelb.edu.au"

__all__ = [
    "CellDiffusion",
    "Unet", 
    "DiffusionSampler",
    "preprocess_adata",
    "postprocess_results"
]




