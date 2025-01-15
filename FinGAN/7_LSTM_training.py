def GradientCheckLSTM(ticker, gen, gen_opt, n_epochs, train_data,batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Gradient check for LSTM-Fin
    """
    ntrain = train_data.shape[0]
    nbatches = ntrain//batch_size+1
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
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            fake = gen(condition,h_0g,c_0g)

            #fake1 = fake1.unsqueeze(0).unsqueeze(2)

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
            STD.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            STD_norm[epoch*nbatches+i] = total_norm

            gen_opt.zero_grad()
            MSE.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            MSE_norm[epoch*nbatches+i] = total_norm

            gen_opt.step()


    alpha = torch.mean(MSE_norm / PnL_norm)
    beta =  0
    gamma =  torch.mean(MSE_norm / SR_norm)
    delta = torch.mean(MSE_norm / STD_norm)
    print("Completed. ")
    print(r"$\alpha$:", alpha)
    print(r"$\beta$:", beta)
    print(r"$\gamma$:", gamma)
    print(r"$\delta$:", delta)

    if plot:


        plt.figure(ticker + " PnL norm")
        plt.title("PnL norm")
        plt.plot(range(len(MSE_norm)),PnL_norm)
        plt.show()

        plt.figure(ticker + " MSE norm")
        plt.title("MSE norm")
        plt.plot(range(len(MSE_norm)), MSE_norm)
        plt.show()

        plt.figure(ticker + " SR norm")
        plt.title("SR norm")
        plt.plot(range(len(MSE_norm)),SR_norm)
        plt.show()

        plt.figure(ticker + " std norm")
        plt.title("std norm")
        plt.plot(range(len(MSE_norm)),STD_norm)
        plt.show()

    return gen, gen_opt, alpha, beta, gamma, delta



def Evaluation2LSTM(ticker,freq,gen,test_data, val_data, h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name, plot = False):
    """
    LSTM(-FIn) evaluation on  a single stock
    """
    df_temp = False
    dt = {'lrd':lrd,'lrg':lrg,'type': losstype,'epochs':n_epochs, 'ticker':ticker}
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
        fake1 = gen(condition1,h0,c0)
        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = fake1[0,:,0]
    mn1 = b1
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
    PnLs = 10000 * np.sign(np.array(ft1.detach())) * np.array(rl1.detach())
    PnLd = np.zeros(int(0.5*len(PnLs)))
    PnL_even = np.zeros(int(0.5*len(PnLs)))
    PnL_odd = np.zeros(int(0.5*len(PnLs)))
    for i1 in range(len(PnLd)):
        PnLd[i1] = PnLs[2*i1] + PnLs[2*i1+1]
        PnL_even[i1] = PnLs[2*i1]
        PnL_odd[i1] = PnLs[2 * i1 + 1]
    PnL1 = np.mean(PnLd)
    #print("PnL in bp", PnL)
    dt['PnL_m test'] = np.mean(PnLd)
    PnL_test = PnLd

    dt['SR_m scaled test'] = np.sqrt(252) * np.mean(PnLd) / np.std(PnLd)


    print("Annualised (test) SR_m: ", np.sqrt(252) * np.mean(PnLd) / np.std(PnLd))

    if (ntest % 2) == 0:
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    else:
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr'] = np.corrcoef([means,reals])[0,1]
    print('Correlation ', np.corrcoef([means,reals])[0,1])
    dt['Pos mn'] = np.sum(means >0)/ len(means)
    dt['Neg mn'] = np.sum(means <0)/ len(means)
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
        fake1 = gen(condition1,h0,c0)

        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = fake1[0,:,0]
    mn1 = b1
    real1 = val_data[:,-1]
    rl1 = real1.squeeze()
    rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
    mae1 = torch.mean(torch.abs(mn1-rl1))
    #print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE val'] = rmse1.item()
    dt['MAE val'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    PnLs = 10000 * np.sign(np.array(ft1.detach())) * np.array(rl1.detach())
    PnLd = np.zeros(int(0.5*len(PnLs)))
    for i1 in range(len(PnLd)):
        PnLd[i1] = PnLs[2*i1] + PnLs[2*i1+1]
    PnL1 = np.mean(PnLd)
    #print("PnL in bp", PnL)
    dt['PnL_m val'] = PnL1

    dt['SR_m scaled val'] = np.sqrt(252) * np.mean(PnLd) / np.std(PnLd)



    print("Annualised (val) SR_m : ", np.sqrt(252 * freq) * getSR(ft1,rl1).item())
    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr val'] = np.corrcoef([means,reals])[0,1]
    dt['Pos mn val'] = np.sum(means >0)/ len(means)
    dt['Neg mn val'] = np.sum(means <0)/ len(means)
    df_temp = pd.DataFrame(data=dt,index=[0])
    return df_temp, PnL_test, PnL_even, PnL_odd