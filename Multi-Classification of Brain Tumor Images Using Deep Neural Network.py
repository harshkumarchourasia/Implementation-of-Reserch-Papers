'''I am using only the first dataset provided in the paper 
"Multi-Classification of Brain Tumor Images Using Deep Neural Network"
as the second dataset is of 10 GB.
The Dataset described here is acquired from Nanfang Hospital and General Hospital,
Tianjing Medical University, China from 2005 to 2010 ,
and then published online with various versions since
2015 reaching to its last release in 2017.
The dataset can be downloaded from https://figshare.com/articles/brain_tumor_dataset/1512427/5
To preprocess and reproduce the dataset as described in this notebook please refer to the steps given here https://github.com/guillaumefrd/brain-tumor-mri-dataset
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout2d
from torch.optim import Adam, SGD
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from skimage.transform import rotate, AffineTransform, warp
import skimage.io as io
import ctypes
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
labels = np.load('brain_tumor_dataset/labels.npy')
images = np.load('brain_tumor_dataset/images.npy')
masks = np.load('brain_tumor_dataset/masks.npy')
labels = [x-1 for x in labels] 
# the labels for the three classes are 1, 2 and 3, But to use the softmax function of pytorch, it must be converted into 0, 1, 2
labels = np.array(labels)
# function to add salt noise to the image the takes 2-d array as input
def SaltNoise(img, prob=0.1):
    shape = img.shape
    noise = np.random.rand(*shape)
    img_sn = img.copy()
    img_sn[noise <  prob] = 1
    return img_sn
# function to resize the image into 128 X 128 
def resize(img):
    return cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
# vector X and Y contains all the training images and labels of the augmentatied dataset
train_img = []
Y = []
for itr in tqdm(range(len(labels))): # //5 should not be here, but np.array(train_img) fails without it
    # adding the orignal image
    train_img.append(resize(images[itr,:,:]))
    Y.append(labels[itr])
    
    # rotating the image by 45 degrees
    train_img.append(resize(rotate(images[itr,:,:], angle=45, mode = 'wrap')))
    Y.append(labels[itr])
     
    # flipped around x-axis
    train_img.append(resize(cv2.rotate(images[itr,:,:], cv2.ROTATE_180)))
    Y.append(labels[itr])
    
    #flipped around y-axis
    train_img.append(resize(np.fliplr(images[itr,:,:])))
    Y.append(labels[itr])
    
    #adding salt noise to the image
    train_img.append(resize(SaltNoise(images[itr,:,:])))
    Y.append(labels[itr])
train_img = np.array(train_img)
Y = np.array(Y)
#X = np.array(train_img)
#Y = np.array(Y)
train_x, val_x, train_y, val_y = train_test_split(train_img,Y, test_size = 0.32)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
# As described in the paper 68% of the dataset is used for training and the rest for validation and testing
# converting training images into torch format
train_x = train_x.reshape(train_x.shape[0], 1, 128, 128)
#train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int);
#train_y = torch.from_numpy(train_y)
# shape of training data
print("shape of training data is",train_x.shape, train_y.shape)

# converting validation images into torch format
val_x = val_x.reshape(val_x.shape[0], 1, 128, 128)
#val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int);
#val_y = torch.from_numpy(val_y)
 
# shape of validation data
print("shape of validation data is", val_x.shape," ", val_y.shape)
# These classes helps to load data and implement batch wise training
class TrainHelper(Dataset):
    def __init__(self):
        self.len = train_x.shape[0]
        self.train_x, self.train_y = train_x.astype('float32'), train_y.astype('float32')
        self.train_x = torch.from_numpy(train_x)
        self.train_y = torch.from_numpy(train_y)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.train_x[idx], self.train_y[idx]
class ValHelper(Dataset):
    def __init__(self):
        self.len = val_x.shape[0]
        self.val_x, self.val_y = val_x.astype('float32'), val_y.astype('float32')
        self.val_x = torch.from_numpy(val_x)
        self.val_y = torch.from_numpy(val_y)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.val_x[idx], self.val_y[idx]
traindataset = TrainHelper()
valdataset = ValHelper()
train_loader = DataLoader(dataset = traindataset, batch_size=32,shuffle=True,num_workers=4)
val_loader = DataLoader(dataset = valdataset, batch_size=32,shuffle=True,num_workers=4)
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 64, kernel_size=10, stride=1, padding=0),
            ReLU(inplace=True),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(0.10),
            
            Conv2d(128, 128, kernel_size=2, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),         
            Dropout2d(0.20)            
        )

        self.linear_layers = Sequential(
            Linear(32768,3)   # softmax layer
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)   #fc layer
        x = self.linear_layers(x)
        return x
# defining the model
model = Net()
# defining the optimizer
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.01)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
print(model)
ctr1=0
ctr2=0
def train(epoch,tb):
    global ctr1
    global ctr2
    # ctr1 and ctr2 are used to count the epoch for the test and training mini-batches
    model.train()
    for batch in train_loader:
        data, target = batch
        x_train, y_train = Variable(data), Variable(target)
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
        output_train = model(x_train.float())
        loss_train = criterion(output_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        ctr1+=1
        if ctr1%50==0:
            print("epoch",ctr1,"train_loss",loss_train.item())
            tb.add_scalar("Training Loss", loss_train.item(), ctr1)
            correct = 0
            total = 0
            _, predicted = torch.max(output_train.data, 1)
            total = y_train.size(0)
            correct = (predicted == y_train).sum().item()
            tb.add_scalar("Training accuracy", correct/total,ctr1)     
            print("accuracy on training dataset is", correct/total)
    print('#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*')
    for batch in val_loader:
        data, target = batch
        x_val, y_val = Variable(data), Variable(target)
        if torch.cuda.is_available():
            x_val = x_val.cuda()
            y_val = y_val.cuda()
        output_val = model(x_val.float())
        loss_val = criterion(output_val, y_val)
        ctr2+=1
        if ctr2%50==0:
            print("epoch",ctr2,"val_loss",loss_val.item())
            tb.add_scalar("Validation Loss", loss_val.item(), ctr2)
            correct = 0
            total = 0
            _, predicted = torch.max(output_val.data, 1)
            total = y_val.size(0)
            correct = (predicted == y_val).sum().item()
            tb.add_scalar("Validation accuracy", correct/total,ctr2)
            print("accuracy on validation dataset is", correct/total)
n_epochs = 300
tb = SummaryWriter()
for epoch in range(n_epochs):
    train(epoch,tb)
tb.close()
# run on terminal to see the graphs
# tensorboard --logdir=runs
