def FinGAN_combos(ticker,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later = True, lrg = 0.0001, lrd = 0.0001, n_epochs = 500, ngrad = 100, h = 1, l = 10, pred = 1, ngpu = 1, tanh_coeff = 100, tr = 0.8, vl = 0.1, z_dim = 32, hid_d = 64, hid_g = 8, checkpoint_epoch = 20, batch_size = 100, diter = 1, plot = False, freq = 2):
    """
    FinGAN: looking at all combinations, performance on both validation and test set for all
    """
    #initialise the networks first:
    datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}
    results_df = pd.DataFrame(data=datastart)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if ticker[0] == 'X':
        train_data,val_data,test_data, dates_dt = split_train_val_testraw(ticker, dataloc, tr, vl, h, l, pred, plotcheck = False)
    else:
        train_data,val_data,test_data, dates_dt = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plotcheck = False)
    data_tt = torch.from_numpy(train_data)
    train_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(test_data)
    test_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(val_data)
    validation_data = data_tt.to(torch.float).to(device)
    ntest = test_data.shape[0]
    condition_size = l
    target_size = pred
    ref_mean = torch.mean(train_data[0:batch_size,:])
    ref_std = torch.std(train_data[0:batch_size,:])
    discriminator_indim = condition_size+target_size

    gen = Generator(noise_dim=z_dim,cond_dim=condition_size, hidden_dim=hid_g,output_dim=pred,mean =ref_mean,std=ref_std)
    gen.to(device)

    disc = Discriminator(in_dim=discriminator_indim, hidden_dim=hid_d,mean=ref_mean,std=ref_std)
    disc.to(device)

    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lrd)

    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    gen, disc, gen_opt, disc_opt, alpha, beta, gamma, delta = GradientCheck(ticker, gen, disc, gen_opt, disc_opt, criterion, ngrad, train_data,batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)

    f_name = modelsloc + ticker + "-Fin-GAN-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg-"
    f_name1 = ticker + "-Fin-GAN-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    PnL_test = [False] * 10
    print("PnL")
    losstype = "PnL"
    genPnL, discPnL, gen_optPnL, disc_optPnL = TrainLoopMainPnLnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnL.state_dict()}, f_name + "PnL_generator_checkpoint.pth")
    df_temp, PnL_test[0], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnL,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)

    pd.DataFrame(PnL_test[0]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")
    plt.figure("Cummulative PnL "+ticker)
    plt.title("Cummulative PnL "+ticker)
    plt.grid(b = True)
    plt.xlabel("date")
    plt.xticks(rotation=45)
    plt.ylabel("bpts")
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[0]), label = "PnL")
    plt.legend(loc='best')

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.title("Intraday cummulative PnL "+ticker)
        plt.grid(b=True)
        plt.xlabel("date")
        plt.xticks(rotation=45)

        plt.ylabel("bpts")
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = "PnL")
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.title("Overnight cummulative PnL "+ticker)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.grid(b=True)

        plt.ylabel("bpts")
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = "PnL")
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.title("Overnight cummulative PnL "+ticker)
        plt.grid(b=True)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.ylabel("bpts")

        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = "PnL")
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.title("Intraday cummulative PnL "+ticker)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.grid(b=True)
        plt.ylabel("bpts")

        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = "PnL")
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.title("Simulated distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True, label = "PnL")
    plt.xlabel("excess return")
    plt.ylabel("density")
    plt.grid(b=True)
    plt.legend(loc='best')
    plt.axvline(rl_test, color='k', linestyle='dashed', linewidth = 2)

    plt.figure("Means "+ticker)
    plt.title("Simulated means "+ticker)
    plt.hist(reals_test, alpha = 0.6, bins = 100,density = True, stacked=True, label = "True")
    plt.hist(means_gen,alpha = 0.5, bins=100, density = True, stacked=True,label = "PnL")
    plt.xlabel("excess return")
    plt.ylabel("density")
    plt.legend(loc='best')
    plt.grid(b=True)


    print("PnL MSE")
    losstype = "PnL MSE"
    genPnLMSE, discPnLMSE, gen_optPnLMSE, disc_optPnLMSE = TrainLoopMainPnLMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSE.state_dict()}, f_name + "PnLMSE_generator_checkpoint.pth")
    df_temp,  PnL_test[1], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)

    pd.DataFrame(PnL_test[1]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')


    print("PnL MSE STD")
    losstype = "PnL MSE STD"
    genPnLMSESTD, discPnLMSESTD, gen_optPnLMSESTD, disc_optPnLMSESTD = TrainLoopMainPnLMSESTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLMSESTD.state_dict()}, f_name + "PnLMSESTD_generator_checkpoint.pth")
    df_temp,  PnL_test[2], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLMSESTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)

    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[2]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[2]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')

    print("PnL MSE SR")
    losstype = "PnL MSE SR"
    genPnLMSESR, discPnLMSESR, gen_optPnLMSESR, disc_optPnLMSESR = TrainLoopMainPnLMSESRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLMSESR.state_dict()}, f_name + "PnLMSESR_generator_checkpoint.pth")
    df_temp,  PnL_test[3], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLMSESR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[3]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[3]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")


    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')

    print("PnL SR")
    losstype = "PnL SR"
    genPnLSR, discPnLSR, gen_optPnLSR, disc_optPnLSR = TrainLoopMainPnLSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSR_generator_checkpoint.pth")
    df_temp,  PnL_test[4], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[4]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[4]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50, density = True,stacked=True,label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100, density = True,stacked=True,label = losstype)
    plt.legend(loc='best')

    print("PnL STD")
    losstype = "PnL STD"
    genPnLSTD, discPnLSTD, gen_optPnLSTD, disc_optPnLSTD = TrainLoopMainPnLSTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSTD_generator_checkpoint.pth")
    df_temp,  PnL_test[5], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLSTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[5]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[5]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    print("SR")
    losstype = "SR"
    genSR, discSR, gen_optSR, disc_optSR = TrainLoopMainSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "SR_generator_checkpoint.pth")
    df_temp,  PnL_test[6], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[6]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[6]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    print("SR MSE")
    losstype = "SR MSE"
    genSRMSE, discSRMSE, gen_optSRMSE, disc_optSRMSE = TrainLoopMainSRMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genSRMSE.state_dict()}, f_name + "SRMSE_generator_checkpoint.pth")
    df_temp,  PnL_test[7], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genSRMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[7]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[7]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50, density = True,stacked=True,label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')

    print("MSE")
    losstype = "MSE"
    genMSE, discMSE, gen_optMSE, disc_optMSE = TrainLoopMainMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genMSE.state_dict()}, f_name + "MSE_generator_checkpoint.pth")
    df_temp,  PnL_test[8], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[8]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[8]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    print("ForGAN")
    losstype = "ForGAN"
    genfg, discfg, gen_optfg, disc_optfg = TrainLoopForGAN(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genfg.state_dict()}, f_name + "ForGAN_generator_checkpoint.pth")
    df_temp,  PnL_test[9], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genfg,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "BCE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[9]), label = losstype)
    plt.legend(loc='best')
    plt.savefig(plotsloc+ticker+"-FinGAN-CummPnL.png")
    plt.show()
    pd.DataFrame(PnL_test[9]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-intradaycummPnL.png")
        plt.legend(loc='best')
        plt.show()

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-overnightcummPnL.png")
        plt.legend(loc='best')
        plt.show()
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-overnightcummPnL.png")
        plt.legend(loc='best')
        plt.show()

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-intradaycummPnL.png")
        plt.legend(loc='best')
        plt.show()

    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50, density = True,stacked=True, label = losstype)
    plt.savefig(plotsloc+ticker+"-FinGAN-sample-dist.png")
    plt.legend(loc='best')
    plt.show()

    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.savefig(plotsloc+ticker+"-FinGAN-means.png")
    plt.legend(loc='best')
    plt.show()

    corr_m = np.corrcoef(PnL_test)

    # can return tge best (validation) generator here too

    return results_df,corr_m
