import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainpath = 'E:/my_python/dataset/FASHIONMNIST/traindata'
testpath = 'E:/my_python/dataset/FASHIONMNIST/testdata'
EPOCH = 10
BATCH_SIZE = 32
LR = 0.01
embedding_dim = 28
squence_dim = 28
HIDDEN_DIM = 10
NUM_CLASS = 10
N_LSTM_LATER = 2
# get data

def RBG2GRAY(image):
    image = image.convert('L')
    return image

traindata = ImageFolder(root=trainpath,
                        transform=transforms.Compose([
                            RBG2GRAY,
                            transforms.ToTensor()
                        ]))

traindata_loader = DataLoader(
    traindata,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_class, n_layers):
        super(RNN, self).__init__()
        self.hidden_d = hidden_dim
        self.layer_n = n_layers
        self.lstm  = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)# for img embeds == img_size
        self.linear = nn.Linear(hidden_dim, n_class)

    def weight_init(self, bs):
        h0 = torch.zeros(self.layer_n, bs, self.hidden_d).to(device)
        c0 = torch.zeros(self.layer_n, bs, self.hidden_d).to(device)

        return h0, c0

    def forward(self, x):
        self.batch_size = x.size(0)
        weight =self.weight_init(self.batch_size)
        output, _ = self.lstm(x, weight)
        output = self.linear(output[:,-1,:])

        return output

model = RNN(embedding_dim, HIDDEN_DIM, NUM_CLASS, N_LSTM_LATER).to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr= LR)

for epoch in range(EPOCH):
    for image, label in traindata_loader:
        image = torch.from_numpy(np.array(image))
        image = image.reshape(-1, squence_dim, embedding_dim).to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch%1 == 0:
            print('{}/{} Loss:{:.4f}'.format(epoch, EPOCH, loss.item()))
