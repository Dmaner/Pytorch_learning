'''
The implement of GAN
'''
import torch
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image

# Hyperparameter
BATCH_SIZE = 64
EPOCH = 10
IMAGE_SIZE = 28*28
CHANNEL = 1
HIDDEN_SIZE = 256
LR = 0.001
LATENT_SIZE = 64
train_data_path = 'FASHIONMNIST/traindata'
test_data_path = 'FASHIONMNIST/testdata'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# writer = SummaryWriter('logdir')

classes = { 0: 'T-shirt_or_top',
            1:	'Trouser',
            2:	'Pullover',
            3:	'Dress',
            4:	'Coat',
            5:	'Sandal',
            6:	'Shirt',
            7:	'Sneaker',
            8:	'Bag',
            9:	'Ankle boot'}

class FASHIONMNIST(Dataset):

    def __init__(self, data_path, transform = None):
        super(FASHIONMNIST,self) .__init__()
        self.transform = transform
        self.root = data_path
        self._find_class()
        self.samples = self._make_dataset()

    def _find_class(self):
        classes = [d.name for d in os.scandir(self.root)]
        classes.sort()
        self.classes = classes
        self.classes_id = {classes[i]: i for i in range(len(classes))}

    def _make_dataset(self):
        images = []
        for target in sorted(self.classes_id.keys()):
            d = os.path.join(self.root, target)
            for root,_,filenames in sorted(os.walk(d)):
                for filename in filenames:
                    path = os.path.join(root, filename)
                    item = (path, target)
                    images.append(item)

        return images

    def __getitem__(self, item):
        sample = self.samples[item]
        path, label = sample
        label = self.classes_id[label]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        return img ,label

    def __len__(self):
        return len(self.samples)

transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])

train_data = FASHIONMNIST(
    data_path=train_data_path,
    transform=transform
)
test_data = FASHIONMNIST(
    data_path=test_data_path,
    transform=transform
)

traindataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
testdataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE
)



# Discriminator
class D(nn.Module):
    def __init__(self,CHANNEL):
        super(D,self).__init__()
        self.channel = CHANNEL
        self.layers = nn.Sequential(
            nn.Conv2d(self.channel, 64, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(64*12*12, 1),
            nn.Sigmoid(),)
    def forward(self, x):
        output = self.layers(x)
        output = output.view(output.shape[0], -1)
        output = self.linear_layer(output)

        return output

#Generator
class G(nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, IMAGE_SIZE*CHANNEL),
            nn.Tanh()
        )

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        output = self.layers(output)
        return output

Disicriminator = D(CHANNEL).to(device)
Generator = G().to(device)

def reset_grad(D_optimizer,G_optimizer):
    G_optimizer.zero_grad()
    D_optimizer.zero_grad()

criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(Generator.parameters(),lr=LR)
D_optimizer = torch.optim.Adam(Disicriminator.parameters(),lr=LR)

total_step = len(traindataloader)

for epoch in range(EPOCH):
    for i , (imgs, _) in enumerate(traindataloader):
        # set label
        real_label = torch.ones(BATCH_SIZE,1).to(device)
        fake_label = torch.zeros(BATCH_SIZE,1).to(device)

        #==Train the discriminator==#
        # compute the loss of identification of real image
        output = Disicriminator(imgs.to(device))
        d_real_loss = criterion(output, real_label)
        real_score = output

        # compute the loss of identification of fake image
        z = torch.randn(BATCH_SIZE,LATENT_SIZE).to(device)
        fake_img = Generator(z).reshape(BATCH_SIZE,CHANNEL,28,28)
        output_2 = Disicriminator(fake_img)
        d_fake_loss = criterion(output_2, fake_label)
        fake_score = output_2

        # Back propagation
        d_loss = d_real_loss + d_fake_loss
        reset_grad(D_optimizer,G_optimizer)
        d_loss.backward()
        D_optimizer.step()

        #==Train the Generator==#
        # compute the rate of producing real image
        z_2 = torch.randn(BATCH_SIZE,LATENT_SIZE).to(device)
        fake_output = Generator(z_2).reshape(BATCH_SIZE,CHANNEL,28,28)
        label = Disicriminator(fake_output)
        g_loss = criterion(label, real_label)

        # Back propagation
        reset_grad(D_optimizer, G_optimizer)
        g_loss.backward()
        G_optimizer.step()

        # print the loss
        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, EPOCH, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))
    # Save the model checkpoints
    torch.save(Generator.state_dict(), 'model/G.ckpt')
    torch.save(Disicriminator.state_dict(), 'model/D.ckpt')
