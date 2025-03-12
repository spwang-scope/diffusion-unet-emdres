import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32, dropout=0.2):
        super(UNet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, base_filters, dropout)
        self.encoder2 = self.conv_block(base_filters, base_filters * 2, dropout)
        self.encoder3 = self.conv_block(base_filters * 2, base_filters * 4, dropout)
        self.encoder4 = self.conv_block(base_filters * 4, base_filters * 8, dropout)
        
        self.bottleneck = self.conv_block(base_filters * 8, base_filters * 16, dropout)
        
        self.upconv4 = self.upconv(base_filters * 16, base_filters * 8)
        self.decoder4 = self.conv_block(base_filters * 16, base_filters * 8, dropout)
        
        self.upconv3 = self.upconv(base_filters * 8, base_filters * 4)
        self.decoder3 = self.conv_block(base_filters * 8, base_filters * 4, dropout)
        
        self.upconv2 = self.upconv(base_filters * 4, base_filters * 2)
        self.decoder2 = self.conv_block(base_filters * 4, base_filters * 2, dropout)
        
        self.upconv1 = self.upconv(base_filters * 2, base_filters)
        self.decoder1 = self.conv_block(base_filters * 2, base_filters, dropout)
        
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        c1 = self.encoder1(x)
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        
        c2 = self.encoder2(p1)
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        
        c3 = self.encoder3(p2)
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        
        c4 = self.encoder4(p3)
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)
        
        c5 = self.bottleneck(p4)
        
        u6 = self.upconv4(c5)
        u6 = F.interpolate(u6, size=c4.shape[2:], mode='bilinear', align_corners=True)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.decoder4(u6)
        
        u7 = self.upconv3(c6)
        u7 = F.interpolate(u7, size=c3.shape[2:], mode='bilinear', align_corners=True)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.decoder3(u7)
        
        u8 = self.upconv2(c7)
        u8 = F.interpolate(u8, size=c2.shape[2:], mode='bilinear', align_corners=True)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.decoder2(u8)
        
        u9 = self.upconv1(c8)
        u9 = F.interpolate(u9, size=c1.shape[2:], mode='bilinear', align_corners=True)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.decoder1(u9)

        c9 = self.relu(c9)
        
        output = torch.sigmoid(self.final_conv(c9))
        return output
