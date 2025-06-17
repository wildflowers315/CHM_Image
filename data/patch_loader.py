"""
Module for loading and processing image patches.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import rasterio
from typing import Dict, List, Tuple, Optional, Union

class PatchDataset(Dataset):
    """Dataset for loading image patches."""
    def __init__(
        self,
        patch_dir: Union[str, Path],
        gedi_data: Dict[str, np.ndarray],
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            patch_dir: Directory containing patch files
            gedi_data: Dictionary mapping patch IDs to GEDI height data
            transform: Optional transform to apply to patches
        """
        self.patch_dir = Path(patch_dir)
        self.gedi_data = gedi_data
        self.transform = transform
        
        # Get list of patch files
        self.patch_files = sorted(list(self.patch_dir.glob('patch_*.tif')))
        
        # Filter to only patches with GEDI data
        self.patch_files = [
            f for f in self.patch_files
            if f.stem in self.gedi_data
        ]
    
    def __len__(self) -> int:
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a patch and its corresponding GEDI data.
        
        Args:
            idx: Index of patch to get
            
        Returns:
            Dictionary containing:
                - patch: Image patch tensor [channels, time, height, width]
                - mask: Binary mask where GEDI data exists [height, width]
                - height: GEDI height data [height, width]
        """
        patch_file = self.patch_files[idx]
        patch_id = patch_file.stem
        
        # Load patch data
        with rasterio.open(patch_file) as src:
            patch_data = src.read()
            n_bands = src.count
            time_steps = n_bands // 13  # 13 bands per time step
            
            # Reshape to [bands, time, height, width]
            patch_data = patch_data.reshape(13, time_steps, *patch_data.shape[1:])
        
        # Get GEDI data
        height_data = self.gedi_data[patch_id]
        mask = (height_data > 0).astype(np.float32)
        
        # Convert to tensors
        patch_tensor = torch.from_numpy(patch_data).float()
        mask_tensor = torch.from_numpy(mask).float()
        height_tensor = torch.from_numpy(height_data).float()
        
        # Apply transform if provided
        if self.transform is not None:
            patch_tensor = self.transform(patch_tensor)
        
        return {
            'patch': patch_tensor,
            'mask': mask_tensor,
            'height': height_tensor
        }

def create_patch_dataloader(
    patch_dir: Union[str, Path],
    gedi_data: Dict[str, np.ndarray],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[callable] = None
) -> DataLoader:
    """
    Create a DataLoader for patch data.
    
    Args:
        patch_dir: Directory containing patch files
        gedi_data: Dictionary mapping patch IDs to GEDI height data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        transform: Optional transform to apply to patches
        
    Returns:
        DataLoader for patch data
    """
    dataset = PatchDataset(
        patch_dir=patch_dir,
        gedi_data=gedi_data,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader 