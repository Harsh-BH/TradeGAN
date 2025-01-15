def combine_vectors(x, y,dim=-1):
    '''
    Function for combining two tensors
    '''
    combined = torch.cat([x,y],dim=dim)
    combined = combined.to(torch.float)
    return combined

def getPnL(predicted,real,nsamp):
    """
    PnL per trade given nsamp samples, predicted forecast, real data realisations
    in bpts
    """
    sgn_fake = torch.sign(predicted)
    PnL = torch.sum(sgn_fake*real)
    PnL = 10000*PnL/nsamp
    return PnL

def getSR(predicted,real):
    """
    Sharpe Ratio given forecasts predicted of real (not annualised)
    """
    sgn_fake = torch.sign(predicted)
    SR = torch.mean(sgn_fake * real) / torch.std(sgn_fake * real)
    return SR

def Evaluation2(ticker,freq,gen,test_data, val_data, h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name, plot = False):
    """
    Evaluation of a GAN model on a single stock
    """
    df_temp = False
    dt = {'lrd':lrd,'lrg':lrg,'type': losstype,'epochs':n_epochs, 'ticker':ticker,  'hid_g':hid_g, 'hid_d':hid_d}
    #print("Validation set best PnL (in bp): ",PnL_best)
    #print("Checkpoint epoch: ",checkpoint_last_epoch+1)
    ntest = test_data.shape[0]
    gen.eval()
    with torch.no_grad():
        condition1 = test_data[:,0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)
        ntest = test_data.shape[0]
        h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
        fake1 = gen(fake_noise,condition1,h0,c0)
        fake1 = fake1.unsqueeze(0).unsqueeze(2)
        generated1 = torch.empty([1,1,1,ntest,1000])
        generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
        #generated1 = fake1.detach()
        for i in range(999):
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            #print(fake.shape)
            generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
            #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
#             print(generated1.shape)
            del fake1
            del fake_noise
        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = generated1.squeeze()
    mn1 = torch.mean(b1,dim=1)
    real1 = test_data[:,-1]
    rl1 = real1.squeeze()
    rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
    mae1 = torch.mean(torch.abs(mn1-rl1))
    #print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE'] = rmse1.item()
    dt['MAE'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    PnL1 = getPnL(ft1,rl1,ntest)
    #print("PnL in bp", PnL)

    #look at the Sharpe Ratio
    n_b1 = b1.shape[1]
    PnL_ws1 = torch.empty(ntest)
    for i1 in range(ntest):
        fk1 = b1[i1,:]
        pu1 = (fk1>=0).sum()
        pu1 = pu1/n_b1
        pd1 = 1-pu1
        PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
        PnL_ws1[i1] = PnL_temp1.item()
    PnL_ws1 = np.array(PnL_ws1)
    PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
    PnL_even = np.zeros(int(0.5 * len(PnL_ws1)))
    PnL_odd = np.zeros(int(0.5 * len(PnL_ws1)))
    for i1 in range(len(PnL_wd1)):
        PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
        PnL_even[i1] = PnL_ws1[2 * i1]
        PnL_odd[i1] = PnL_ws1[2 * i1 + 1]
    PnL_test = PnL_wd1
    PnL_w_m1 = np.mean(PnL_wd1)
    PnL_w_std1 = np.std(PnL_wd1)
    SR1 = PnL_w_m1/PnL_w_std1
    #print("Sharpe Ratio: ",SR)
    dt['SR_w scaled'] = SR1*np.sqrt(252)
    dt['PnL_w'] = PnL_w_m1

    if (ntest % 2) == 0:
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    else:
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    print("Annualised (test) SR_w: ",SR1*np.sqrt(252))

    distcheck = np.array(b1[1,:].cpu())
    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr'] = np.corrcoef([means,reals])[0,1]
    dt['Pos mn'] = np.sum(means >0)/ len(means)
    dt['Neg mn'] = np.sum(means <0)/ len(means)
    print('Correlation ',np.corrcoef([means,reals])[0,1] )

    dt['narrow dist'] = (np.std(distcheck)<0.0002)

    means_gen = means
    reals_test = reals
    distcheck_test = distcheck
    rl_test = reals[1]

    mn = torch.mean(b1,dim=1)
    mn = np.array(mn.cpu())
    dt['narrow means dist'] = (np.std(mn)<0.0002)

    ntest = val_data.shape[0]
    gen.eval()
    with torch.no_grad():
        condition1 = val_data[:,0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)
        ntest = val_data.shape[0]
        h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
        fake1 = gen(fake_noise,condition1,h0,c0)
        fake1 = fake1.unsqueeze(0).unsqueeze(2)
        generated1 = torch.empty([1,1,1,ntest,1000])
        generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
        #generated1 = fake1.detach()
        for i in range(999):
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            #print(fake.shape)
            generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
            #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
#             print(generated1.shape)
            del fake1
            del fake_noise
        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = generated1.squeeze()
    mn1 = torch.mean(b1,dim=1)
    real1 = val_data[:,-1]
    rl1 = real1.squeeze()
    rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
    mae1 = torch.mean(torch.abs(mn1-rl1))
    #print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE val'] = rmse1.item()
    dt['MAE val'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    #print("PnL in bp", PnL)

    #look at the Sharpe Ratio
    n_b1 = b1.shape[1]
    PnL_ws1 = torch.empty(ntest)
    for i1 in range(ntest):
        fk1 = b1[i1,:]
        pu1 = (fk1>=0).sum()
        pu1 = pu1/n_b1
        pd1 = 1-pu1
        PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
        PnL_ws1[i1] = PnL_temp1.item()
    PnL_ws1 = np.array(PnL_ws1)
    PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
    for i1 in range(len(PnL_wd1)):
        PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
    PnL_w_m1 = np.mean(PnL_wd1)
    PnL_w_std1 = np.std(PnL_wd1)
    SR1 = PnL_w_m1/PnL_w_std1
    #print("Sharpe Ratio: ",SR)
    dt['PnL_w val'] = PnL_w_m1
    dt['SR_w scaled val'] = SR1*np.sqrt(252)

    print("Annualised (val) SR_w : ",SR1*np.sqrt(252))

    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr val'] = np.corrcoef([means,reals])[0,1]
    dt['Pos mn val'] = np.sum(means >0)/ len(means)
    dt['Neg mn val'] = np.sum(means <0)/ len(means)

    df_temp = pd.DataFrame(data=dt,index=[0])

    return df_temp, PnL_test, PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test

def Evaluation3(tickers,freq,gen,test, val, h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name, plot = False):
    """
    Evaluation of a GAN model in the universality setting (multiple tickers)
    """
    df_temp = False
    dt = {'lrd':[],'lrg':[],'type': [],'epochs':[], 'ticker':[],  'hid_g':[], 'hid_d':[]}
    results_df = pd.DataFrame(data = dt)
    PnLs_test = np.zeros((len(tickers), int(0.5 * test[0].shape[0])))
    PnLs_val = np.zeros((len(tickers), int(0.5 * val[0].shape[0])))
    means_test = np.zeros((len(tickers), test[0].shape[0]))
    means_val = np.zeros((len(tickers), val[0].shape[0]))
    # print(means_test.shape)
    #print("Validation set best PnL (in bp): ",PnL_best)
    #print("Checkpoint epoch: ",checkpoint_last_epoch+1)
    for ii in tqdm(range(len(tickers))):
        val_data = val[ii]
        test_data = test[ii]
        ticker = tickers[ii]
        dt = {'lrd':lrd,'lrg':lrg,'type': losstype,'epochs':n_epochs, 'ticker':ticker,  'hid_g':hid_g, 'hid_d':hid_d}
        ntest = test_data.shape[0]
        gen.eval()
        with torch.no_grad():
            condition1 = test_data[:,0:l]
            condition1 = condition1.unsqueeze(0)
            condition1 = condition1.to(device)
            condition1 = condition1.to(torch.float)
            ntest = test_data.shape[0]
            h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            generated1 = torch.empty([1,1,1,ntest,1000])
            generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
            #generated1 = fake1.detach()
            for i in range(999):
                fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
                fake1 = gen(fake_noise,condition1,h0,c0)
                fake1 = fake1.unsqueeze(0).unsqueeze(2)
                #print(fake.shape)
                generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
                #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
    #             print(generated1.shape)
                del fake1
                del fake_noise
            #rmse = torch.sqrt(torch.mean((fake-real)**2))
            #mae = torch.mean(torch.abs(fake-real))
        #print("RMSE: ", rmse)
        #print("MAE: ",mae)
        b1 = generated1.squeeze()
        mn1 = torch.mean(b1,dim=1)
        # print(mn1.shape)
        means_test[ii, :] = np.array(mn1.detach())
        real1 = test_data[:,-1]
        rl1 = real1.squeeze()
        rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
        mae1 = torch.mean(torch.abs(mn1-rl1))
        #print("RMSE: ",rmse,"MAE: ",mae)
        dt['RMSE'] = rmse1.item()
        dt['MAE'] = mae1.item()
        ft1 = mn1.clone().detach().to(device)        #print("PnL in bp", PnL)

        #look at the Sharpe Ratio
        n_b1 = b1.shape[1]
        PnL_ws1 = torch.empty(ntest)
        for i1 in range(ntest):
            fk1 = b1[i1,:]
            pu1 = (fk1>=0).sum()
            pu1 = pu1/n_b1
            pd1 = 1-pu1
            PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
            PnL_ws1[i1] = PnL_temp1.item()
        PnL_ws1 = np.array(PnL_ws1)
        PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
        for i1 in range(len(PnL_wd1)):
            PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
        PnLs_test[ii, :] = PnL_wd1
        PnL_w_m1 = np.mean(PnL_wd1)
        PnL_w_std1 = np.std(PnL_wd1)
        SR1 = PnL_w_m1/PnL_w_std1
        #print("Sharpe Ratio: ",SR)
        dt['PnL_w'] = PnL_w_m1
        dt['SR_w scaled'] = SR1 * np.sqrt(252)
        # print("Annualised (test) SR_w: ",SR1*np.sqrt(252 * freq))
        # print("Annualised (test) SR_m: ", np.sqrt(252 * freq) * getSR(ft1,rl1).item())
        dist_loc = plotsloc+"distcheck-"+f_name+".png"

        distcheck = np.array(b1[1,:].cpu())
        means = np.array(mn1.detach())
        reals = np.array(rl1.detach())
        dt['Corr'] = np.corrcoef([means,reals])[0,1]
        dt['Pos mn'] = np.sum(means >0)/ len(means)
        dt['Neg mn'] = np.sum(means <0)/ len(means)
        # print('Correlation ',np.corrcoef([means,reals])[0,1] )

        dt['narrow dist'] = (np.std(distcheck)<0.0002)

        means_loc = plotsloc+"recovered-means-"+f_name+".png"


        mn = torch.mean(b1,dim=1)
        mn = np.array(mn.cpu())
        dt['narrow means dist'] = (np.std(mn)<0.0002)


        ntest = val_data.shape[0]
        gen.eval()
        with torch.no_grad():
            condition1 = val_data[:,0:l]
            condition1 = condition1.unsqueeze(0)
            condition1 = condition1.to(device)
            condition1 = condition1.to(torch.float)
            ntest = val_data.shape[0]
            h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            generated1 = torch.empty([1,1,1,ntest,1000])
            generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
            #generated1 = fake1.detach()
            for i in range(999):
                fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
                fake1 = gen(fake_noise,condition1,h0,c0)
                fake1 = fake1.unsqueeze(0).unsqueeze(2)
                #print(fake.shape)
                generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
                #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
    #             print(generated1.shape)
                del fake1
                del fake_noise
            #rmse = torch.sqrt(torch.mean((fake-real)**2))
            #mae = torch.mean(torch.abs(fake-real))
        #print("RMSE: ", rmse)
        #print("MAE: ",mae)
        b1 = generated1.squeeze()
        mn1 = torch.mean(b1,dim=1)
        means_val[ii, :] = np.array(mn1.detach())

        real1 = val_data[:,-1]
        rl1 = real1.squeeze()
        rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
        mae1 = torch.mean(torch.abs(mn1-rl1))
        #print("RMSE: ",rmse,"MAE: ",mae)
        dt['RMSE val'] = rmse1.item()
        dt['MAE val'] = mae1.item()
        ft1 = mn1.clone().detach().to(device)
        #print("PnL in bp", PnL)

        #look at the Sharpe Ratio
        n_b1 = b1.shape[1]
        PnL_ws1 = torch.empty(ntest)
        for i1 in range(ntest):
            fk1 = b1[i1,:]
            pu1 = (fk1>=0).sum()
            pu1 = pu1/n_b1
            pd1 = 1-pu1
            PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
            PnL_ws1[i1] = PnL_temp1.item()
        PnL_ws1 = np.array(PnL_ws1)

        PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
        for i1 in range(len(PnL_wd1)):
            PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
        PnLs_val[ii, :] = PnL_wd1
        PnL_w_m1 = np.mean(PnL_wd1)
        PnL_w_std1 = np.std(PnL_wd1)
        SR1 = PnL_w_m1/PnL_w_std1
        #print("Sharpe Ratio: ",SR)
        dt['PnL_w val'] = PnL_w_m1
        dt['SR_w scaled val'] = SR1*np.sqrt(freq)


        # print("Annualised (val) SR_w : ",SR1*np.sqrt(252 * freq))
        # print("Annualised (val) SR_m : ", np.sqrt(252 * freq) * getSR(ft1,rl1).item())

        means = np.array(mn1.detach())
        reals = np.array(rl1.detach())
        dt['Corr val'] = np.corrcoef([means,reals])[0,1]
        dt['Pos mn val'] = np.sum(means >0)/ len(means)
        dt['Neg mn val'] = np.sum(means <0)/ len(means)
        df_temp = pd.DataFrame(data=dt,index=[0])
        results_df = pd.concat([results_df,df_temp], ignore_index=True)
    PnL_test = np.sum(PnLs_test,axis=0)
    PnL_val = np.sum(PnLs_val,axis=0)

    return results_df, PnL_test, PnL_val, means_test, means_val

def GradientCheck(ticker, gen, disc, gen_opt, disc_opt, criterion, n_epochs, train_data,batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Gradient norm check
    """
    ntrain = train_data.shape[0]
    nbatches = ntrain//batch_size+1
    BCE_norm = torch.empty(nbatches*n_epochs, device = device)
    PnL_norm = torch.empty(nbatches*n_epochs, device = device)
    MSE_norm = torch.empty(nbatches*n_epochs, device = device)
    SR_norm = torch.empty(nbatches*n_epochs, device = device)
    STD_norm = torch.empty(nbatches*n_epochs, device = device)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]

    #currstep = 0
    #train the discriminator more

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

            # Update generator
            # Zero out the generator gradients


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
            MSE = (torch.norm(ft-rl)**2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))
            STD = torch.std(PnL_s)
            gen_opt.zero_grad()
            SR.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            SR_norm[epoch*nbatches+i] = total_norm

            gen_opt.zero_grad()
            PnL.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            PnL_norm[epoch*nbatches+i] = total_norm

            gen_opt.zero_grad()
            MSE.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            MSE_norm[epoch*nbatches+i] = total_norm

            gen_opt.zero_grad()
            STD.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            STD_norm[epoch*nbatches+i] = total_norm

            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            BCE_norm[epoch*nbatches+i] = total_norm
            gen_opt.step()


    alpha = torch.mean(BCE_norm / PnL_norm)
    beta =  torch.mean(BCE_norm / MSE_norm)
    gamma =  torch.mean(BCE_norm / SR_norm)
    delta = torch.mean(BCE_norm / STD_norm)
    print("Completed. ")
    print(r"$\alpha$:", alpha)
    print(r"$\beta$:", beta)
    print(r"$\gamma$:", gamma)
    print(r"$\delta$:", delta)

    if plot:
        plt.figure(ticker + " BCE norm")
        plt.title(ticker + " BCE norm")
        plt.plot(range(len(BCE_norm)),BCE_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()

        plt.figure(ticker + " PnL norm")
        plt.title(ticker +" PnL norm")
        plt.plot(range(len(BCE_norm)),PnL_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()

        plt.figure(ticker + " MSE norm")
        plt.title(ticker + " MSE norm")
        plt.plot(range(len(BCE_norm)), MSE_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()

        plt.figure(ticker + " SR norm")
        plt.title("SR norm")
        plt.plot(range(len(BCE_norm)),SR_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()

        plt.figure(ticker + " STD norm")
        plt.title(ticker + " STD norm")
        plt.plot(range(len(BCE_norm)),STD_norm)
        plt.ylabel(r"$L^2$ norm")
        plt.xlabel("iteration")
        plt.show()

        # plt.figure(ticker + " Norms")
        # plt.title(ticker + " gradient norms")
        # plt.plot(range(len(BCE_norm)),BCE_norm, label = "BCE")
        # plt.plot(range(len(BCE_norm)),PnL_norm, label = "PnL")
        # plt.plot(range(len(BCE_norm)),SR_norm, label = "SR")
        # plt.plot(range(len(BCE_norm)),STD_norm, label = "STD")
        # plt.ylabel(r"$L^2$ norm")
        # plt.xlabel("iteration")
        # plt.legend(loc = 'best')
        # plt.show()


    return gen, disc, gen_opt, disc_opt, alpha, beta, gamma, delta