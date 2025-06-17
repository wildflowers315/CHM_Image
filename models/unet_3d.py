"""
3D U-Net model implementation for canopy height mapping.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DownBlock3D(nn.Module):
    """Downsampling block with maxpool and convolutions."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_block = ConvBlock3D(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x

class UpBlock3D(nn.Module):
    """Upsampling block with transposed convolution."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )
        self.conv_block = ConvBlock3D(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle cases where input dimensions don't match exactly
        diff_y = skip.size(3) - x.size(3)
        diff_x = skip.size(4) - x.size(4)
        
        x = F.pad(x, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2,
            0, 0  # No padding in temporal dimension
        ])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x

class UNet3D(nn.Module):
    """3D U-Net model for volumetric segmentation."""
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        base_channels: int = 64,
        depth: int = 4
    ):
        super().__init__()
        self.depth = depth
        
        # Initial convolution block
        self.inc = ConvBlock3D(in_channels, base_channels)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(depth):
            out_channels = current_channels * 2
            self.down_blocks.append(DownBlock3D(current_channels, out_channels))
            current_channels = out_channels
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            in_channels = current_channels
            out_channels = current_channels // 2
            self.up_blocks.append(UpBlock3D(in_channels, out_channels))
            current_channels = out_channels
        
        # Final convolution
        self.outc = nn.Conv3d(base_channels, n_classes, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        """Initialize model weights."""
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x1 = self.inc(x)
        
        # Downsampling path
        x_down = [x1]
        for i in range(self.depth):
            x_down.append(self.down_blocks[i](x_down[-1]))
        
        # Upsampling path
        x = x_down[-1]
        for i in range(self.depth):
            x = self.up_blocks[i](x, x_down[-(i+2)])
        
        # Final convolution
        x = self.outc(x)
        
        # Average over temporal dimension
        x = torch.mean(x, dim=2)
        
        return x

def create_3d_unet(
    in_channels: int,
    n_classes: int,
    base_channels: int = 64,
    depth: int = 4
) -> UNet3D:
    """
    Create a 3D U-Net model.
    
    Args:
        in_channels: Number of input channels
        n_classes: Number of output classes
        base_channels: Number of channels in first layer
        depth: Number of downsampling/upsampling steps
        
    Returns:
        Initialized 3D U-Net model
    """
    model = UNet3D(
        in_channels=in_channels,
        n_classes=n_classes,
        base_channels=base_channels,
        depth=depth
    )
    return model 