import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data_csv = 'data/fashion_mnist/fashion-mnist_test.csv'
train_data_csv = 'data/fashion_mnist/fashion-mnist_train.csv'

learning_rate = 0.001
num_epoch = 80
IMAGE_SIZE = 28
CHANNEL = 1
BATCH_SIZE = 100

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(4), # Arrange image size to 28 to 32
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
])

class MyDataset(data.Dataset):
    '''
    Build your own dataset
    '''
    def __init__(self, Data_csv_file, transform = None):
        self.fashion_mnist = list(pd.read_csv(Data_csv_file).values)
        self.transform = transform
        label, img = [],[]
        for one_line in self.fashion_mnist:
            label.append(one_line[0])
            img.append(one_line[1:])
        self.label = np.asarray(label)
        self.img = np.asarray(img).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL).astype('float32')

    def __getitem__(self, item):
        label, img = self.label[item], self.img[item]
        if self.transform is not None:
            img = self.transform(img)

        return  img,label

    def __len__(self):
        return len(self.label)

def con3x3(inchannel, outchannel, stride=1):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            con3x3(inchannels, outchannels, stride),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            con3x3(outchannels, outchannels),
            nn.BatchNorm2d(outchannels)
        )
        if stride != 1 or outchannels != inchannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels,
                          outchannels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(outchannels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual_output = self.layer(x)
        ouput = residual_output + self.shortcut(x)
        ouput = F.relu(ouput)

        return ouput


class My_resnet18(nn.Module):
    def __init__(self, ResidualBlock, prechannel,layers, num_class=10):
        super(My_resnet18, self).__init__()
        self.channel = 64

        # 1 conv + 4*basicblock(contain 4 layers) + 1 fc
        self.conv1 = nn.Sequential(
            nn.Conv2d(prechannel, 64, kernel_size=3, stride=1, padding=1, bias=1),
            nn.BatchNorm2d(64)
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_class)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.channel, channels, stride))
            self.channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output,4)

        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(criterion, learning_rate, optimizer):

    total_step = len(train_data_loader)

    for epoch in range(num_epoch):
        for i,(img, label) in enumerate(train_data_loader):
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print('Epoch {}/{} Step {}/{} Loss: {:.4f}'.format(
                    epoch, num_epoch, i+1, total_step, loss.item()
                ))
        if (epoch+1)%20 == 0:
            learning_rate /= 3
            update_lr(optimizer, learning_rate)

train_data = MyDataset(train_data_csv,transform)
test_data = MyDataset(test_data_csv,transform)

train_data_loader = data.DataLoader(dataset=train_data,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)
test_data_loader = data.DataLoader(dataset=test_data,
                                   batch_size=BATCH_SIZE)

model = My_resnet18(ResidualBlock, 1, [2,2,2,2], num_class=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(criterion, learning_rate, optimizer)

#test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')