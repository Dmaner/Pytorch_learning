'''
References: https://github.com/milesial/Pytorch-UNet
Train dataset: BSD 500
'''
import torch.nn as nn
import torch
import torch.nn.functional as F

class Double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down_sample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down_sample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class Up_sample(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(Up_sample,self).__init__()
        if bilinear:
            self.layer = nn.Upsample(scale_factor=2,
                                     mode='bilinear',
                                     align_corners=True)
        else:
            self.layer = nn.ConvTranspose2d(in_channel//2,
                                            out_channel//2,
                                            kernel_size=2,
                                            stride=2)

    def forward(self, x1, x2):
        x1 = self.layer(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = Double_conv(n_channels, 64)
        self.down1 = Down_sample(64, 128)
        self.down2 = Down_sample(128, 256)
        self.down3 = Down_sample(256, 512)
        self.down4 = Down_sample(512, 512)
        self.up1 = Up_sample(1024, 256)
        self.up2 = Up_sample(512, 128)
        self.up3 = Up_sample(256, 64)
        self.up4 = Up_sample(128, 64)
        self.outc = nn.Conv2d(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
