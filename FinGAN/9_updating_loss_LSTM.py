import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rnd

def TrainLoopnLSTMPnL(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the PnL loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients

            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            gen_loss = SqLoss - alpha * PnL
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:


        plt.figure("LSTM loss PnL")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMPnLSTD(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the PnL, STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients

            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            STD = torch.std(PnL_s)
            gen_loss = SqLoss - alpha * PnL + delta * STD
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:


        plt.figure("LSTM loss PnL STD")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMPnLSR(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the PnL,SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients

            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SR = torch.mean(PnL_s) / torch.std(PnL_s)
            gen_loss = SqLoss - alpha * PnL - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:

        plt.figure("LSTM loss PnL SR")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMSR(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients

            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            SR = torch.mean(PnL_s) / torch.std(PnL_s)
            gen_loss = SqLoss - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:


        plt.figure("LSTM loss SR")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMSTD(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients

            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            STD = torch.std(PnL_s)
            gen_loss = SqLoss + delta * STD
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:


        plt.figure("LSTM loss STD")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTM(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients

            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            gen_loss = SqLoss
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:


        plt.figure("LSTM loss")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt
