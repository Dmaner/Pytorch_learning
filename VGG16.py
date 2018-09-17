from torchvision import models, transforms
import numpy as np
from PIL import Image
from torch import nn

Vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class Vgg16(nn.Module):
    def __init__(self, layers, num_classes=1000, init_weight=True):
        super(Vgg16, self).__init__()
        self.conv_layers = layers
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weight:
            self.weight_init()


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return  output

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, bn=False):
    layers = nn.ModuleList()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2,2))
        else:
            conv_2d = nn.Conv2d(in_channels, v, kernel_size=3,stride=1,padding=1)
            if bn:
                layers.extend([nn.BatchNorm2d(v),nn.ReLU(True)])
            else:
                layers.extend([conv_2d, nn.ReLU(True)])
            in_channels = v

    return layers

# Test
img = Image.open('E:/my_python/Test/bee_black.png')
print(np.array(img).shape)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
Conv_layers = make_layers(Vgg16_cfg)
model = Vgg16(Conv_layers)
other_model = models.vgg16()
output = model(transform(img).unsqueeze(0))
print(output.shape)# torch.Size([1,1000])
