# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from collections import namedtuple
import pickle
import copy
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_loss(model, loss_func, x, y, train=False):
    if train:
        model.train()
        y_pred = model.forward(x)
        loss = loss_func(y_pred, y)
    else:
        model.eval()
        with torch.no_grad():
            y_pred = model.forward(x)
            loss = loss_func(y_pred, y)
    return loss

def predict(model, x):
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(x).argmax(dim=1)
    return y_pred.cpu().numpy()

Data = namedtuple('Data', ['feature', 'target'])

class RandomSamplingTrainer(object):
    def __init__(self, model, loss_func, optimizer):
        self.model = model #Neural Network
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.n_updates = 0
    
    def set_dataset(self, dataset):
        self.buffer = copy.copy(dataset)

    def train(self, batch_size):
        self.n_updates += 1
        idx_list, data_list = self.sample(batch_size)
        self.update(idx_list, data_list)
      
    def sample(self, batch_size):
        idx_batch = np.random.choice(len(self.buffer), batch_size)
        return idx_batch, [self.buffer[i] for i in idx_batch]
        
    def update(self, idx_list, data_list):
        train_batch = Data(*zip(*data_list))
        features = torch.cat(train_batch.feature).to(device)
        targets = torch.Tensor(train_batch.target).long().to(device)
        loss = compute_loss(self.model, self.loss_func, features, targets, train=True)

        loss_mean = loss.mean()
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


class PrioritizedSamplingTrainer(RandomSamplingTrainer):
    def set_dataset(self, dataset, init_loss=100, epsilon=10**(-4)):
        self.buffer = copy.copy(dataset)
        self.loss_buffer = np.array([init_loss for _ in range(len(dataset))], dtype=np.float)
        self.epsilon = epsilon

    def train(self, batch_size):
        self.n_updates += 1
        idx_list, data_list = self.sample(batch_size)
        loss_list = self.update(idx_list, data_list)
        self.update_loss_buffer(idx_list, loss_list)

    def sample(self, batch_size):
        sum_loss = sum(self.loss_buffer)
        probs = self.loss_buffer / sum_loss
        idx_batch = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        return idx_batch, [self.buffer[i] for i in idx_batch]

    def update_loss_buffer(self, idx_list, loss_list):
        for i, loss in zip(idx_list, loss_list):
            self.loss_buffer[i] = loss + self.epsilon

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.fc = nn.Linear(2*2*64, 200)
        self.out_layer = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        y = F.relu(self.conv1(x))
        y = F.max_pool2d(y, kernel_size=3, stride=3)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2)
        y = y.view(-1, 2*2*64)
        y = F.relu(self.fc(y))
        y = self.out_layer(y)
        return y


if __name__ == '__main__':
    #Set a seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load dataset
    with open('train_data_list.pkl', mode='rb') as f:
        train_dataset_list = pickle.load(f)
    with open('test_data_list.pkl', mode='rb') as f:
        test_dataset_list = pickle.load(f)
    
    test_dataset = Data(*zip(*test_dataset_list))
    test_features = torch.cat(test_dataset.feature).to(device)
    test_targets = torch.Tensor(test_dataset.target).long().to(device)

    sampling_method_list = ['Random', 'Prioritized']
    trainer_object_list = [RandomSamplingTrainer, PrioritizedSamplingTrainer]
    test_loss_traj = {method:[] for method in sampling_method_list}

    for method, traier_object in zip(sampling_method_list, trainer_object_list):

        model = CNN().to(device)
        loss_func = nn.CrossEntropyLoss(reduce=False)
        optimizer = optim.Adam(model.parameters())

        trainer = traier_object(model, loss_func, optimizer)
        trainer.set_dataset(train_dataset_list)
    
        print(method)
        print('---- Start Training ----')
        for _ in range(20000):
            trainer.train(batch_size=120)
            #For varidation
            test_loss = compute_loss(model, loss_func, test_features, test_targets).detach().cpu().numpy().mean()
            test_loss_traj[method].append(test_loss)
            #Output learning status
            if trainer.n_updates % 100 == 0:
                print(f'#updates:{trainer.n_updates} loss:{test_loss}')
        print('---- Finish Training ----')
    
    loss_traj_random_smoothing = [np.mean(test_loss_traj['Random'][max(0, i-99):i]) for i in range(1, 20001)]
    loss_traj_prioritized_smoothing = [np.mean(test_loss_traj['Prioritized'][max(0, i-99):i]) for i in range(1, 20001)]
    plt.plot(np.arange(1, 20001), loss_traj_random_smoothing, label='Random')
    plt.plot(np.arange(1, 20001), loss_traj_prioritized_smoothing, label='Prioritized')
    plt.legend()
    plt.title('Loss for test dataset')
    plt.xlabel('#update')
    plt.ylim(0, 1)
    plt.show()
