import torch
import torch.nn as nn

# 2D U-Net Model for non-temporal data
class Height2DUNet(nn.Module):
    """2D U-Net for canopy height prediction from non-temporal patches."""
    
    def __init__(self, in_channels, n_classes=1, base_channels=64):
        super().__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.decoder4 = self.upconv_block(base_channels * 16, base_channels * 8)
        self.decoder3 = self.upconv_block(base_channels * 16, base_channels * 4)  # 16 = 8 + 8 from skip
        self.decoder2 = self.upconv_block(base_channels * 8, base_channels * 2)   # 8 = 4 + 4 from skip
        self.decoder1 = self.upconv_block(base_channels * 4, base_channels)       # 4 = 2 + 2 from skip
        
        # Final prediction
        self.final_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # (B, 64, H, W)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))  # (B, 128, H/2, W/2)
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))  # (B, 256, H/4, W/4)
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))  # (B, 512, H/8, W/8)
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))  # (B, 1024, H/16, W/16)
        
        # Decoder with skip connections
        d4 = self.decoder4(b)  # (B, 512, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)  # (B, 1024, H/8, W/8)
        
        d3 = self.decoder3(d4)  # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 512, H/4, W/4)
        
        d2 = self.decoder2(d3)  # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 256, H/2, W/2)
        
        d1 = self.decoder1(d2)  # (B, 64, H, W)
        
        # Final prediction
        out = self.final_conv(d1)  # (B, 1, H, W)
        
        return out.squeeze(1)  # (B, H, W)