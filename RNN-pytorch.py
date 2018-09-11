import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 50
LR = 0.005
NUM_CLASS = 10
IMAGE_SIZE = 28
CHANNEL = 1
TRAIN_DATA_FILE_CSV = 'data/fashion_mnist/fashion-mnist_train.csv'
TEST_FILE_CSV = 'data/fashion_mnist/fashion-mnist_test.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_CLOTHING = {0:'T-shirt/top',
                  1:'Trouser',
                  2:'Pullover',
                  3:'Dress',
                  4:'Coat',
                  5:'Sandal',
                  6:'Shirt',
                  7:'Sneaker',
                  8:'Bag',
                  9:'Ankle boot'}

class MyDataset(Dataset):
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

        return label, img

    def __len__(self):
        return len(self.label)