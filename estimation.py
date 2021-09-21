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

torch.set_default_tensor_type(torch.FloatTensor)


class Estim_Net(nn.Module):
    def __init__(self, n_input, n_h, n_output, n_epochs=1000):
        super(Estim_Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_output)
        self.n_epochs = n_epochs

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x_pre_smax = self.fc3(x)
        x = F.softmax(x_pre_smax, dim=1)
        return x, x_pre_smax

    def learn(self, x_train, y, batch_size):
        loss_fn = nn.MSELoss(reduction='mean')
        # optimizer = optim.SGD(self.parameters(), lr=0.1)
        optimizer = optim.AdamW(self.parameters(), lr=0.001)

        for epoch in range(self.n_epochs):
            # no mini-batching
            if batch_size <= 0:
                yhat, _ = self.forward(x_train)
                # error = y_tensor - yhat
                # loss = (error ** 2).mean()
                loss = loss_fn(y, yhat)
            else:    # use mini-batch
                permutation = torch.randperm(x_train.size()[0])
                x_mini = x_train[permutation[:batch_size]]
                yhat, _ = self.forward(x_mini)
                # error = y_tensor - yhat
                # loss = (error ** 2).mean()
                loss = loss_fn(y[permutation[:batch_size]], yhat)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if epoch % (self.n_epochs//10) == 0:
            #     print(f'Loss: {float(loss.data.numpy())}')
