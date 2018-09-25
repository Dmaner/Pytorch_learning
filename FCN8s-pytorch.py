import torch.nn as nn
import torch
import numpy as np

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8s(nn.Module):
    def __init__(self, num_class=21):
        super(FCN8s, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100), #for padding 100 pixels for
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # get 1/2 image
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # get 1/4 image
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # get 1/8 image
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # get 1/16 image
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # get 1/32 image
        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout2d()
        )
        self.score_fr = nn.Conv2d(4096, num_class, 1) # classifier for the pixels
        self.score_pool3 = nn.Conv2d(256, num_class, 1)
        self.score_pool4 = nn.Conv2d(512, num_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            num_class, num_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            num_class, num_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_class, num_class, 4, stride=2, bias=False)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        pool3 = h
        h = self.conv4(h)
        pool4 = h
        h = self.conv5(h)
        h = self.fc(h)
        h = self.score_fr(h)
        upscore2 = self.upscore2(h) # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

a = torch.randn(1, 3, 224, 224)
model = FCN8s()
print(model(a).shape)
# torch.Size(1,21,224,224)

