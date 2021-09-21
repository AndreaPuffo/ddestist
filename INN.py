import timeit

import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from petc_generation import PETC_Generation
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

torch.set_default_tensor_type(torch.FloatTensor)


class INN(nn.Module):
    def __init__(self, n_input, n_h, n_output, n_epochs=1000):
        super(INN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_output)
        self.n_epochs = n_epochs
        self.neg_slope = 0.1

    def forward(self, x):
        """
        classic forward pass
        :param x:
        :return:
        """
        act = torch.nn.LeakyReLU(self.neg_slope)
        z1 = act(self.fc1(x))
        z2 = act(self.fc2(z1))
        x_pre_smax = self.fc3(z2)
        x = F.softmax(x_pre_smax, dim=1)
        return x, x_pre_smax

    def bckwrd(self, y):
        """
        backward pass of the inverse NN
        :param y: pre-softmax vector
        :return:
        """
        pinv_w1 = torch.pinverse(self.fc1.weight.T)
        pinv_w2 = torch.pinverse(self.fc2.weight.T)
        pinv_w3 = torch.pinverse(self.fc3.weight.T)
        b1, b2, b3 = self.fc1.bias, self.fc2.bias, self.fc3.bias
        # actual inverse computation
        z2hat = self.inverse_lr((y - b3) @ pinv_w3)
        z1hat = self.inverse_lr((z2hat - b2) @ pinv_w2)
        xhat = (z1hat - b1) @ pinv_w1
        return xhat

    def inverse_lr(self, x):
        """
        inverse leaky relu function
        :param x:
        :return:
        """
        return torch.relu(x) - 1.0/self.neg_slope * torch.relu(-x)

    def learn(self, x_train, y, batch_size):
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.MSELoss()
        # optimizer = optim.SGD(self.parameters(), lr=0.1)
        optimizer = optim.AdamW(self.parameters(), lr=0.005)

        for epoch in range(self.n_epochs):
            permutation = torch.randperm(x_train.size()[0])
            x_mini = x_train[permutation[:batch_size]]

            yhat, _ = self.forward(x_mini)
            # z2hat, z1hat, xhat = self.bckwrd(y_pre_sm)
            # error = y_tensor - yhat
            # loss = (error ** 2).mean()

            loss = loss_fn(y[permutation[:batch_size]], yhat)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if epoch % (self.n_epochs//10) == 0:
            #     print(f'Loss: {float(loss.data.numpy())}')
