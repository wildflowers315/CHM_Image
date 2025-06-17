import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """Double 3D convolution block with batch normalization and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downsampling block with max pooling and double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upsampling block with transposed convolution and double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle different sizes
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2,
                       diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    """Output convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class Height3DUNet(nn.Module):
    """
    3D U-Net architecture for canopy height prediction.
    
    Args:
        in_channels: Number of input channels (bands)
        n_classes: Number of output classes (default=1 for regression)
        base_channels: Number of base channels in first layer
    """
    
    def __init__(self, in_channels: int, n_classes: int = 1, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        
        # Initial convolution
        self.inc = DoubleConv3D(in_channels, base_channels)
        
        # Downsampling path
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        self.down4 = Down3D(base_channels * 8, base_channels * 16)
        
        # Upsampling path
        self.up1 = Up3D(base_channels * 16, base_channels * 8)
        self.up2 = Up3D(base_channels * 8, base_channels * 4)
        self.up3 = Up3D(base_channels * 4, base_channels * 2)
        self.up4 = Up3D(base_channels * 2, base_channels)
        
        # Output convolution
        self.outc = OutConv3D(base_channels, n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            
        Returns:
            Output tensor [batch, n_classes, height, width]
        """
        # Add temporal dimension if not present
        if len(x.shape) == 4:
            x = x.unsqueeze(2)  # [batch, channels, 1, height, width]
        
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        x = self.outc(x)
        
        # Remove temporal dimension for final output
        x = x.squeeze(2)  # [batch, n_classes, height, width]
        
        return x

def create_3d_unet(
    in_channels: int,
    n_classes: int = 1,
    base_channels: int = 64,
    pretrained: bool = False,
    pretrained_path: str = None
) -> Height3DUNet:
    """
    Create a 3D U-Net model.
    
    Args:
        in_channels: Number of input channels (bands)
        n_classes: Number of output classes (default=1 for regression)
        base_channels: Number of base channels in first layer
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights
        
    Returns:
        Initialized 3D U-Net model
    """
    model = Height3DUNet(
        in_channels=in_channels,
        n_classes=n_classes,
        base_channels=base_channels
    )
    
    if pretrained and pretrained_path:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
    
    return model 