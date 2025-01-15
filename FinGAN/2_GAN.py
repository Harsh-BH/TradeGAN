import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rnd


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, noise_dim,cond_dim, hidden_dim,output_dim,mean,std):
        super(Generator, self).__init__()
        self.input_dim = noise_dim+cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        #predicting a single value, so the output dimension is 1
        self.mean = mean
        self.std = std
        #Add the modules

        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
        # nn.init.xavier_normal_(self.lstm.weight)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.linear1 = nn.Linear(in_features=self.hidden_dim+self.noise_dim, out_features=self.hidden_dim+self.noise_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(in_features=self.hidden_dim+self.noise_dim, out_features=output_dim)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.ReLU()


    def forward(self, noise,condition,h_0,c_0):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''
        #x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        condition = (condition-self.mean)/self.std
        out, (h_n, c_n) = self.lstm(condition, (h_0, c_0))
        out = combine_vectors(noise.to(torch.float),h_n.to(torch.float),dim=-1)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = out*self.std+self.mean
        return out

class LSTM(nn.Module):
    '''
    Values:
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, noise_dim,cond_dim, hidden_dim,output_dim,mean,std):
        super(LSTM, self).__init__()
        self.input_dim = noise_dim+cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        #predicting a single value, so the output dimension is 1
        self.mean = mean
        self.std = std
        #Add the modules

        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=self.output_dim, num_layers=1, dropout=0)
        # nn.init.xavier_normal_(self.lstm.weight)
        # nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        # nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.activation = nn.ReLU()


    def forward(self, condition,h_0,c_0):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''
        #x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        condition = (condition-self.mean)/self.std
        out, (h_n, c_n) = self.lstm(condition, (h_0, c_0))
        out = out*self.std+self.mean
        return out

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      in_dim: the input dimension (noise dim + conditin dim + forecast dim for the condition for this dataset), a scalar
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, in_dim, hidden_dim,mean,std):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.mean = mean
        self.std = std
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        self.sigmoid = nn.Sigmoid()



    def forward(self, in_chan,h_0,c_0):
        '''
        in_chan: concatenated condition with real or fake
        h_0 and c_0: for the LSTM
        '''
        x = in_chan
        x = (x-self.mean)/self.std
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.linear(h_n)
        out = self.sigmoid(out)
        return out