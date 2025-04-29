import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

TrainData = "D:/Competition set/MI_Four_mat/jwkset/S1T.mat"
TestData = "D:/Competition set/MI_Four_mat/jwkset/S1E.mat"
TrainLabel = "D:/Competition set/MI_Four_mat/jwkset/L1T.mat"
TestLabel = "D:/Competition set/MI_Four_mat/jwkset/L1E.mat"

TrainDataSet = sio.loadmat(TrainData)
TestDataSet = sio.loadmat(TestData)
Train_Label = sio.loadmat(TrainLabel)
Test_Label = sio.loadmat(TestLabel)

x_train = TrainDataSet['X_train']
x_test = TestDataSet['X_test']
y_train = Train_Label['Y_train'].T
y_test = Test_Label['Y_test'].T


y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(y_test)




sig_train = x_train
label_train = y_train
sig_test = x_test
label_test = y_test

class Train_Data(Dataset):
    def __init__(self, sig_train, label_train):
        self.sig_train = sig_train
        self.label_train = label_train

    def __getitem__(self, idx):
        return self.sig_train[idx], self.label_train[idx]

    def __len__(self):
        return self.sig_train.shape[0]
        # return len(self.sig_train)


class Test_Data(Dataset):
    def __init__(self, sig_test, label_test):
        self.sig_test = sig_test
        self.label_test = label_test

    def __getitem__(self, idx):
        return self.sig_test[idx], self.label_test[idx]

    def __len__(self):
        return self.sig_test.shape[0]
        # return len(self.sig_test)

class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]
