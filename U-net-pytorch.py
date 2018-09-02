'''
References: https://github.com/milesial/Pytorch-UNet
Train dataset: BSD 500
'''
import torch.nn as nn

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
    def __init__(self, in_ch, out_ch):
        super(Down_sample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class 

class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()

