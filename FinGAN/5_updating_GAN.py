def TrainLoopForGAN(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for the BCE GAN (ForGAN)
    """
    ntrain = train_data.shape[0]
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

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
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
    print("Training completed, checkpoint epoch: ", checkpoint_last_epoch)
    # print("PnL val (best):", PnL_best)
    print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt