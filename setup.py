"""Setup script for CellDiffusion package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))

# Get package version
version = {}
with open(os.path.join(this_directory, "CellDiffusion", "_version.py")) as f:
    exec(f.read(), version)

# Read requirements
with open(os.path.join(this_directory, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="CellDiffusion",
    version=version["__version__"],
    author=version["__author__"],
    author_email=version["__email__"],
    description="A Python package for generating pseudo-cells using diffusion models",
    long_description="""
CellDiffusion is a generalized implementation of diffusion models for single-cell RNA sequencing data. 
It allows users to generate pseudo-cells from AnnData objects, providing a powerful tool for data 
augmentation and synthetic cell generation in computational biology.

Key Features:
- Easy-to-use API with AnnData input/output
- Support for cluster-specific training and generation
- Flexible UNet architecture with configurable parameters
- Multiple noise schedules (linear, cosine, quadratic, sigmoid)
- Automatic preprocessing and postprocessing pipelines
- GPU acceleration support
- Model checkpointing and resuming
""",
    long_description_content_type="text/plain",
    url="https://github.com/yourusername/CellDiffusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "jupyter",
            "matplotlib",
            "seaborn",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-napoleon",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords="single-cell, diffusion-models, pseudo-cells, bioinformatics, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/CellDiffusion/issues",
        "Source": "https://github.com/yourusername/CellDiffusion",
        "Documentation": "https://celldiffusion.readthedocs.io/",
    },
)





