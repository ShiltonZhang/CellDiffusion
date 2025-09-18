"""
Core CellDiffusion class for generating pseudo-cells using diffusion models.

This module contains the main CellDiffusion class that provides a high-level
interface for training diffusion models and generating pseudo-cells from AnnData objects.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import scanpy as sc

from .models import Unet
from .diffusion import DiffusionSampler
from .utils import (
    preprocess_adata, 
    adata_to_tensor, 
    postprocess_results, 
    validate_adata, 
    get_device,
    set_random_seed,
    calculate_gene_padding_size
)


class CellDiffusion:
    """
    CellDiffusion: Generate pseudo-cells using diffusion models.
    
    This class provides a complete pipeline for training diffusion models on single-cell
    RNA sequencing data and generating pseudo-cells. It takes AnnData objects as input
    and returns AnnData objects with generated pseudo-cells annotated in the observations.
    
    Args:
        image_size (int): Size to reshape gene expression data (image_size x image_size)
        timesteps (int): Number of diffusion timesteps
        beta_schedule (str): Type of beta schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
        dim (int): Base dimension for UNet model
        dim_mults (tuple): Dimension multipliers for UNet
        channels (int): Number of channels (usually 1 for gene expression)
        device (str, optional): Device to use ('cuda' or 'cpu'). Auto-detected if None
        random_seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        image_size: int = 48,
        timesteps: int = 500,
        beta_schedule: str = 'linear',
        dim: int = 48,
        dim_mults: tuple = (1, 2, 4, 8),
        channels: int = 1,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        # Set random seed
        set_random_seed(random_seed)
        
        # Configuration
        self.image_size = image_size
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.dim = dim
        self.dim_mults = dim_mults
        self.channels = channels
        self.device = device or get_device()
        self.random_seed = random_seed
        
        # Initialize diffusion sampler
        self.diffusion_sampler = DiffusionSampler(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            device=self.device
        )
        
        # Model and training state
        self.model = None
        self.optimizer = None
        self.training_history = []
        self.is_trained = False
        
        # Data storage
        self.original_adata = None
        self.preprocessed_adata = None
        self.training_data = None
        self.cluster_models = {}  # For cluster-specific models
        
        print(f"CellDiffusion initialized on device: {self.device}")
    
    def preprocess(
        self,
        adata,
        min_counts: int = 3,
        target_sum: float = 1e4,
        n_top_genes: Optional[int] = None,
        log_transform: bool = True,
        highly_variable: bool = True,
        copy: bool = False
    ):
        """
        Preprocess AnnData object for diffusion model training.
        
        Args:
            adata: Input AnnData object
            min_counts: Minimum counts for gene filtering
            target_sum: Target sum for normalization
            n_top_genes: Number of highly variable genes (auto-calculated if None)
            log_transform: Whether to apply log transformation
            highly_variable: Whether to select highly variable genes
            copy: Whether to copy the input data
            
        Returns:
            AnnData: Preprocessed AnnData object
        """
        validate_adata(adata)
        
        # Auto-calculate n_top_genes if not provided
        if n_top_genes is None:
            n_top_genes = self.image_size * self.image_size
            print(f"Auto-setting n_top_genes to {n_top_genes} (image_size^2)")
        
        # Store original data
        self.original_adata = adata.copy() if not copy else adata
        
        # Preprocess
        self.preprocessed_adata = preprocess_adata(
            adata,
            min_counts=min_counts,
            target_sum=target_sum,
            n_top_genes=n_top_genes,
            log_transform=log_transform,
            highly_variable=highly_variable,
            copy=True
        )
        
        print(f"Preprocessing complete. Shape: {self.preprocessed_adata.shape}")
        return self.preprocessed_adata
    
    def _create_model(self):
        """Create UNet model."""
        self.model = Unet(
            dim=self.dim,
            channels=self.channels,
            dim_mults=self.dim_mults
        ).to(self.device)
        
        return self.model
    
    def fit(
        self,
        adata = None,
        cluster_key: Optional[str] = None,
        cluster_value: Optional[str] = None,
        epochs: int = 300,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        loss_type: str = 'huber',
        save_every: int = 50,
        save_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Train the diffusion model on the provided data.
        
        Args:
            adata: AnnData object (uses preprocessed if None)
            cluster_key: Key in adata.obs for cluster-specific training
            cluster_value: Specific cluster to train on
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            loss_type: Loss function ('l1', 'l2', 'huber')
            save_every: Save model every N epochs
            save_path: Path to save model checkpoints
            verbose: Whether to print training progress
            
        Returns:
            dict: Training history
        """
        # Use preprocessed data if no adata provided
        if adata is None:
            if self.preprocessed_adata is None:
                raise ValueError("No data provided and no preprocessed data available. Call preprocess() first.")
            adata = self.preprocessed_adata
        
        # Convert to tensor format
        data_tensor, cell_indices = adata_to_tensor(
            adata,
            image_size=self.image_size,
            cluster_key=cluster_key,
            cluster_value=cluster_value
        )
        
        print(f"Training on {data_tensor.shape[0]} cells")
        print(f"Data tensor shape: {data_tensor.shape}")
        
        # Create model and optimizer
        if self.model is None:
            self._create_model()
        
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Create data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.training_history = []
        self.model.train()
        
        if verbose:
            print("Starting training...")
        
        for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
            epoch_losses = []
            
            for step, (batch,) in enumerate(dataloader):
                loss = self.diffusion_sampler.train_step(
                    self.model, batch, self.optimizer, loss_type=loss_type
                )
                epoch_losses.append(loss)
                
                if verbose and step % 100 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            
            # Record average loss for epoch
            avg_loss = np.mean(epoch_losses)
            self.training_history.append(avg_loss)
            
            # Save checkpoint
            if save_path and (epoch + 1) % save_every == 0:
                self.save_model(save_path, epoch)
        
        self.is_trained = True
        
        if verbose:
            print(f"Training complete! Final loss: {self.training_history[-1]:.4f}")
        
        return {
            'history': self.training_history,
            'final_loss': self.training_history[-1] if self.training_history else None,
            'epochs': epochs
        }
    
    def fit_by_clusters(
        self,
        adata = None,
        cluster_key: str = 'leiden',
        epochs: int = 300,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        loss_type: str = 'huber',
        save_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Train separate models for each cluster.
        
        Args:
            adata: AnnData object (uses preprocessed if None)
            cluster_key: Key in adata.obs for cluster information
            epochs: Number of training epochs per cluster
            batch_size: Training batch size
            learning_rate: Learning rate
            loss_type: Loss function
            save_path: Base path to save models
            verbose: Whether to print progress
            
        Returns:
            dict: Training results for each cluster
        """
        if adata is None:
            if self.preprocessed_adata is None:
                raise ValueError("No data provided and no preprocessed data available. Call preprocess() first.")
            adata = self.preprocessed_adata
        
        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
        
        clusters = adata.obs[cluster_key].unique()
        results = {}
        
        if verbose:
            print(f"Training models for {len(clusters)} clusters")
        
        for cluster in tqdm(clusters, desc="Clusters", disable=not verbose):
            if verbose:
                print(f"\nTraining model for cluster {cluster}")
            
            # Create new model for this cluster
            cluster_model = Unet(
                dim=self.dim,
                channels=self.channels,
                dim_mults=self.dim_mults
            ).to(self.device)
            
            # Prepare cluster-specific data
            data_tensor, cell_indices = adata_to_tensor(
                adata,
                image_size=self.image_size,
                cluster_key=cluster_key,
                cluster_value=cluster
            )
            
            if data_tensor.shape[0] == 0:
                if verbose:
                    print(f"No cells found for cluster {cluster}, skipping")
                continue
            
            # Train model
            optimizer = Adam(cluster_model.parameters(), lr=learning_rate)
            dataset = TensorDataset(data_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            history = []
            cluster_model.train()
            
            for epoch in range(epochs):
                epoch_losses = []
                for batch, in dataloader:
                    loss = self.diffusion_sampler.train_step(
                        cluster_model, batch, optimizer, loss_type=loss_type
                    )
                    epoch_losses.append(loss)
                
                avg_loss = np.mean(epoch_losses)
                history.append(avg_loss)
                
                if verbose and epoch % 50 == 0:
                    print(f"Cluster {cluster}, Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Store model and results
            self.cluster_models[str(cluster)] = cluster_model
            results[str(cluster)] = {
                'model': cluster_model,
                'history': history,
                'final_loss': history[-1] if history else None,
                'n_cells': data_tensor.shape[0]
            }
            
            # Save model if path provided
            if save_path:
                cluster_save_path = f"{save_path}_cluster_{cluster}.pth"
                torch.save(cluster_model.state_dict(), cluster_save_path)
        
        if verbose:
            print("Cluster-specific training complete!")
        
        return results
    
    def generate(
        self,
        n_samples: int = 1000,
        cluster_key: Optional[str] = None,
        cluster_value: Optional[str] = None,
        batch_size: int = 500,
        use_original_data: bool = False,
        progress: bool = True
    ):
        """
        Generate pseudo-cells using the trained model.
        
        Args:
            n_samples: Number of pseudo-cells to generate
            cluster_key: Cluster key for cluster-specific generation
            cluster_value: Specific cluster to generate from
            batch_size: Batch size for generation
            use_original_data: Whether to start from original data points
            progress: Whether to show progress bar
            
        Returns:
            AnnData: AnnData object with generated pseudo-cells
        """
        if not self.is_trained and not self.cluster_models:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Select model to use
        if cluster_value and str(cluster_value) in self.cluster_models:
            model = self.cluster_models[str(cluster_value)]
            if progress:
                print(f"Using cluster-specific model for cluster {cluster_value}")
        elif self.model is not None:
            model = self.model
        else:
            raise ValueError("No trained model available")
        
        model.eval()
        
        # Generate in batches
        all_generated = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        if progress:
            print(f"Generating {n_samples} pseudo-cells in {n_batches} batches")
        
        for i in tqdm(range(n_batches), desc="Generating", disable=not progress):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            
            if use_original_data and self.preprocessed_adata is not None:
                # Start from original data points
                data_tensor, _ = adata_to_tensor(
                    self.preprocessed_adata,
                    image_size=self.image_size,
                    cluster_key=cluster_key,
                    cluster_value=cluster_value
                )
                
                if data_tensor.shape[0] > 0:
                    # Sample random subset
                    indices = np.random.choice(data_tensor.shape[0], current_batch_size, replace=True)
                    batch_data = data_tensor[indices]
                    generated = self.diffusion_sampler.sample_from_data(model, batch_data, progress=False)
                else:
                    # Fall back to random generation
                    generated = self.diffusion_sampler.sample(
                        model, self.image_size, current_batch_size, self.channels, progress=False
                    )
                    generated = generated[-1]  # Get final result
            else:
                # Generate from random noise
                generated = self.diffusion_sampler.sample(
                    model, self.image_size, current_batch_size, self.channels, progress=False
                )
                generated = generated[-1]  # Get final result
            
            all_generated.append(generated)
        
        # Combine all generated data
        all_generated = np.concatenate(all_generated, axis=0)[:n_samples]
        
        # Convert back to AnnData format
        if self.original_adata is not None:
            result_adata = postprocess_results(
                all_generated,
                self.original_adata,
                combine_with_original=True,
                image_size=self.image_size
            )
        else:
            raise ValueError("No original AnnData available for postprocessing")
        
        # Add cluster information if specified
        if cluster_value is not None:
            pseudo_mask = result_adata.obs['is_pseudo'] == True
            result_adata.obs.loc[pseudo_mask, f'generated_from_cluster'] = str(cluster_value)
        
        if progress:
            print(f"Generated {n_samples} pseudo-cells")
            print(f"Final dataset shape: {result_adata.shape}")
        
        return result_adata
    
    def save_model(self, path: str, epoch: Optional[int] = None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(path)
        if epoch is not None:
            save_path = save_path.parent / f"{save_path.stem}_epoch_{epoch}{save_path.suffix}"
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'training_history': self.training_history,
            'config': {
                'image_size': self.image_size,
                'timesteps': self.timesteps,
                'beta_schedule': self.beta_schedule,
                'dim': self.dim,
                'dim_mults': self.dim_mults,
                'channels': self.channels
            }
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load configuration
        config = checkpoint.get('config', {})
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Create and load model
        self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        self.training_history = checkpoint.get('training_history', [])
        self.is_trained = True
        
        print(f"Model loaded from {path}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'image_size': self.image_size,
            'timesteps': self.timesteps,
            'beta_schedule': self.beta_schedule,
            'dim': self.dim,
            'dim_mults': self.dim_mults,
            'channels': self.channels,
            'device': self.device,
            'is_trained': self.is_trained,
            'n_cluster_models': len(self.cluster_models)
        }




