"""
Diffusion process and sampling functionality for CellDiffusion package.

This module contains the diffusion model implementation including forward diffusion,
reverse sampling, and various noise schedules.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    
    Args:
        timesteps (int): Number of diffusion timesteps
        s (float): Small offset to prevent beta from being too small near t=0
        
    Returns:
        torch.Tensor: Beta schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """
    Linear beta schedule.
    
    Args:
        timesteps (int): Number of diffusion timesteps
        
    Returns:
        torch.Tensor: Beta schedule
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    """
    Quadratic beta schedule.
    
    Args:
        timesteps (int): Number of diffusion timesteps
        
    Returns:
        torch.Tensor: Beta schedule
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """
    Sigmoid beta schedule.
    
    Args:
        timesteps (int): Number of diffusion timesteps
        
    Returns:
        torch.Tensor: Beta schedule
    """
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    """
    Extract values from tensor a at indices t and reshape for broadcasting.
    
    Args:
        a (torch.Tensor): Tensor to extract from
        t (torch.Tensor): Indices to extract
        x_shape (tuple): Shape to broadcast to
        
    Returns:
        torch.Tensor: Extracted and reshaped values
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionSampler:
    """
    Diffusion sampler for generating pseudo-cells.
    
    This class handles the diffusion process including forward diffusion (adding noise)
    and reverse sampling (denoising) using a trained UNet model.
    
    Args:
        timesteps (int): Number of diffusion timesteps
        beta_schedule (str): Type of beta schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    
    def __init__(self, timesteps=500, beta_schedule='linear', device='cpu'):
        self.timesteps = timesteps
        self.device = device
        
        # Create beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            self.betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-calculate useful values
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_start (torch.Tensor): Clean input data
            t (torch.Tensor): Timestep indices
            noise (torch.Tensor, optional): Noise to add
            
        Returns:
            torch.Tensor: Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        """
        Calculate loss for training the denoising model.
        
        Args:
            denoise_model (nn.Module): The UNet denoising model
            x_start (torch.Tensor): Clean input data
            t (torch.Tensor): Timestep indices
            noise (torch.Tensor, optional): Noise to add
            loss_type (str): Type of loss ('l1', 'l2', 'huber')
            
        Returns:
            torch.Tensor: Computed loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t).
        
        Args:
            model (nn.Module): The trained UNet model
            x (torch.Tensor): Noisy data at timestep t
            t (torch.Tensor): Current timestep
            t_index (int): Current timestep index
            
        Returns:
            torch.Tensor: Denoised data at timestep t-1
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Use the model to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, img=None, progress=True):
        """
        Complete reverse diffusion sampling loop.
        
        Args:
            model (nn.Module): The trained UNet model
            shape (tuple): Shape of the samples to generate
            img (torch.Tensor, optional): Starting image (if None, start from noise)
            progress (bool): Whether to show progress bar
            
        Returns:
            list: List of images at each timestep (optional)
            torch.Tensor: Final generated sample
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise or provided image
        if img is None:
            img = torch.randn(shape, device=device)
        
        imgs = []
        iterator = reversed(range(0, self.timesteps))
        if progress:
            iterator = tqdm(iterator, desc='Sampling', total=self.timesteps)
        
        for i in iterator:
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1, progress=True):
        """
        Generate samples from the trained model.
        
        Args:
            model (nn.Module): The trained UNet model
            image_size (int): Size of the square images to generate
            batch_size (int): Number of samples to generate
            channels (int): Number of channels
            progress (bool): Whether to show progress bar
            
        Returns:
            list: Generated samples at each timestep
        """
        return self.p_sample_loop(
            model, 
            shape=(batch_size, channels, image_size, image_size),
            progress=progress
        )
    
    @torch.no_grad()
    def sample_from_data(self, model, data, progress=True):
        """
        Generate samples starting from specific data points.
        
        Args:
            model (nn.Module): The trained UNet model
            data (torch.Tensor): Input data to start sampling from
            progress (bool): Whether to show progress bar
            
        Returns:
            torch.Tensor: Final generated samples
        """
        device = next(model.parameters()).device
        data = data.to(device)
        
        # Add noise to the maximum timestep
        t = torch.tensor([self.timesteps - 1], device=device)
        t = t.repeat(data.shape[0])
        noisy_data = self.q_sample(data, t)
        
        # Sample from noisy data
        imgs = self.p_sample_loop(model, data.shape, img=noisy_data, progress=progress)
        
        # Return the final denoised result
        return imgs[-1] if imgs else noisy_data.cpu().numpy()
    
    def train_step(self, model, batch, optimizer, loss_type="huber"):
        """
        Single training step for the diffusion model.
        
        Args:
            model (nn.Module): The UNet model
            batch (torch.Tensor): Training batch
            optimizer (torch.optim.Optimizer): Optimizer
            loss_type (str): Type of loss to use
            
        Returns:
            float: Loss value
        """
        optimizer.zero_grad()
        
        batch_size = batch.shape[0]
        batch = batch.to(self.device)
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Calculate loss
        loss = self.p_losses(model, batch, t, loss_type=loss_type)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()




