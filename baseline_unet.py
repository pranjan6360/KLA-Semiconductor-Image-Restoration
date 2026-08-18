import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Convolutional Block with LeakyReLU activation.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class BaselineSRUNet(nn.Module):
    """
    Lightweight 2x Super-Resolution & Denoising UNet Architecture for Semiconductor Inspection.
    
    Input Shape:  (B, 1, 128, 128) [Noisy Low-Resolution Image]
    Output Shape: (B, 1, 256, 256) [Restored High-Resolution Image]
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64):
        super(BaselineSRUNet, self).__init__()
        
        # 1. Initial Convolution (Feature Extraction)
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # 2. Encoder
        self.enc1 = ResBlock(num_features) # (B, 64, 128, 128)
        self.down1 = nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=2, padding=1) # (B, 128, 64, 64)
        
        self.enc2 = ResBlock(num_features * 2) # (B, 128, 64, 64)
        
        # 3. Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(num_features * 2),
            ResBlock(num_features * 2)
        )
        
        # 4. Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2) # (B, 64, 128, 128)
        self.dec1_conv = nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1)
        self.dec1_res = ResBlock(num_features) # (B, 64, 128, 128)
        
        # 5. 2x Learnable Super-Resolution Upsampling (PixelShuffle)
        self.upsample_conv = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1) # (B, 256, 128, 128)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2) # (B, 64, 256, 256)
        
        # 6. Final Convolution to Output Channel
        self.tail = nn.Sequential(
            ResBlock(num_features),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Initial Feature Extraction
        head_feat = self.head(x) # (B, 64, 128, 128)
        
        # Encoder Stage 1
        e1 = self.enc1(head_feat) # (B, 64, 128, 128)
        d1 = self.down1(e1) # (B, 128, 64, 64)
        
        # Encoder Stage 2 & Bottleneck
        e2 = self.enc2(d1) # (B, 128, 64, 64)
        b = self.bottleneck(e2) # (B, 128, 64, 64)
        
        # Decoder Stage 1 with Skip Connection
        u1 = self.up1(b) # (B, 64, 128, 128)
        cat1 = torch.cat([u1, e1], dim=1) # (B, 128, 128, 128)
        d1_feat = self.dec1_res(self.dec1_conv(cat1)) + head_feat # (B, 64, 128, 128)
        
        # 2x Learnable Super-Resolution Upsampling
        up_feat = self.upsample_conv(d1_feat) # (B, 256, 128, 128)
        sr_feat = self.pixel_shuffle(up_feat) # (B, 64, 256, 256)
        
        # Final Reconstruction
        out = self.tail(sr_feat) # (B, 1, 256, 256)
        
        # Output Constrained to [0, 1]
        out_clipped = torch.clamp(out, 0.0, 1.0)
        return out_clipped
