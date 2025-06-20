"""Data loaders for different training scenarios."""

import torch
from torch.utils.data import DataLoader, random_split
from typing import List, Tuple, Optional

from .datasets import AugmentedPatchDataset, SparseGEDIDataset


def create_patch_dataloader(patch_files: List[str],
                           batch_size: int = 8,
                           validation_split: float = 0.2,
                           augment: bool = True,
                           augment_factor: int = 12,
                           num_workers: int = 0,
                           random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for patch-based models.
    
    Args:
        patch_files: List of patch file paths
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        augment: Whether to apply data augmentation
        augment_factor: Number of augmentation combinations
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set random seed for reproducible splits
    torch.manual_seed(random_seed)
    
    # Split patch files
    n_patches = len(patch_files)
    n_val = int(n_patches * validation_split)
    n_train = n_patches - n_val
    
    train_files, val_files = random_split(
        patch_files, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Convert back to lists
    train_files = [patch_files[i] for i in train_files.indices]
    val_files = [patch_files[i] for i in val_files.indices]
    
    # Create datasets
    train_dataset = AugmentedPatchDataset(
        train_files,
        augment=augment,
        augment_factor=augment_factor,
        validation_mode=False
    )
    
    val_dataset = AugmentedPatchDataset(
        val_files,
        augment=False,  # No augmentation for validation
        augment_factor=1,
        validation_mode=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader


def create_sparse_dataloader(patch_files: List[str],
                            batch_size: int = 1024,
                            validation_split: float = 0.2,
                            min_gedi_pixels: int = 10,
                            num_workers: int = 0,
                            random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for sparse GEDI pixel training (RF, MLP).
    
    Args:
        patch_files: List of patch file paths
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        min_gedi_pixels: Minimum GEDI pixels required per patch
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = SparseGEDIDataset(
        patch_files,
        min_gedi_pixels=min_gedi_pixels
    )
    
    if len(full_dataset) == 0:
        raise ValueError("No valid GEDI samples found in patches")
    
    # Split dataset
    torch.manual_seed(random_seed)
    n_samples = len(full_dataset)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader


def get_dataloader_info(loader: DataLoader) -> dict:
    """Get information about a data loader."""
    dataset = loader.dataset
    
    info = {
        'num_samples': len(dataset),
        'batch_size': loader.batch_size,
        'num_batches': len(loader),
        'num_workers': loader.num_workers,
        'dataset_type': type(dataset).__name__
    }
    
    # Add dataset-specific info
    if hasattr(dataset, 'augment_factor'):
        info['augment_factor'] = dataset.augment_factor
        info['augmentation_enabled'] = dataset.augment
    
    if hasattr(dataset, 'patch_files'):
        info['num_patch_files'] = len(dataset.patch_files)
    
    return info


def calculate_steps_per_epoch(loader: DataLoader) -> int:
    """Calculate number of steps per epoch for a data loader."""
    return len(loader)


def estimate_training_time(train_loader: DataLoader,
                          val_loader: Optional[DataLoader],
                          epochs: int,
                          seconds_per_step: float = 1.0) -> dict:
    """
    Estimate training time based on data loader sizes.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        seconds_per_step: Estimated seconds per training step
        
    Returns:
        Dictionary with time estimates
    """
    train_steps = len(train_loader)
    val_steps = len(val_loader) if val_loader else 0
    
    steps_per_epoch = train_steps + val_steps
    total_steps = steps_per_epoch * epochs
    
    estimated_seconds = total_steps * seconds_per_step
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60
    
    return {
        'train_steps_per_epoch': train_steps,
        'val_steps_per_epoch': val_steps,
        'total_steps_per_epoch': steps_per_epoch,
        'total_epochs': epochs,
        'total_steps': total_steps,
        'estimated_seconds': estimated_seconds,
        'estimated_minutes': estimated_minutes,
        'estimated_hours': estimated_hours,
        'formatted_time': f"{int(estimated_hours)}h {int(estimated_minutes % 60)}m"
    }