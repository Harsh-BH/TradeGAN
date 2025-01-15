def TrainLoopMainPnLnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot=False):
    """
    Training loop: PnL and MSE loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more
    SR_best = 0
    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + beta * SqLoss
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss


    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed ")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLMSESRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL, MSE, SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + beta * SqLoss - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLMSESTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL, MSE, STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + beta * SqLoss + delta * torch.std(PnL_s)
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    checkpoint_last_epoch = 0
    SR_best = 0
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)

    return gen, disc, gen_opt, disc_opt

def TrainLoopMainMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: MSE loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))  + beta * SqLoss
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainSRMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: SR, MSE loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) + beta * SqLoss - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss


    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLSTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL, STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    checkpoint_last_epoch = 0
    SR_best = 0
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
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)

                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)


            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            STD = torch.std(PnL_s)

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + delta * STD
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()

        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)

    return gen, disc, gen_opt, disc_opt