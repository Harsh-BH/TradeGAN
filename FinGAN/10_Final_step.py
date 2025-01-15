def FinGAN_universal(tickers1, other,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later = True, lrg = 0.0001, lrd = 0.0001, n_epochs = 500, ngrad = 100, h = 1, l = 10, pred = 1, ngpu = 1, tanh_coeff = 100, tr = 0.8, vl = 0.1, z_dim = 32, hid_d = 64, hid_g = 8, checkpoint_epoch = 20, batch_size = 100, diter = 1, plot = False, freq = 2):
    """
    FinGAN loss combos in the universal setting
    """
    #initialise the networks first:
    datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}
    results_df = pd.DataFrame(data=datastart)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    ticker = tickers1[0]
    train_data,val_data,test_data, dates_dt = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plot)
    ntr = train_data.shape[0]
    nvl = val_data.shape[0]
    ntest = test_data.shape[0]
    n_tickers1 = len(tickers1)
    n_tickers = len(tickers1) + len(other)
    train_data = np.zeros((ntr * n_tickers1, l + pred))
    validation_data = [False] * n_tickers
    test_data = [False] * n_tickers
    for i in range(n_tickers1):
        ticker = tickers1[i]
        if ticker[0] == "X":
            train,val,test, _ = split_train_val_testraw(ticker, dataloc,  tr, vl, h, l, pred, plot)
        else:
            train,val,test, _ = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plot)
        data_tt = torch.from_numpy(test)
        test_data[i] = data_tt.to(torch.float).to(device)
        train_data[i*ntr:(i+1)*ntr] = train
        data_tt = torch.from_numpy(val)
        validation_data[i] = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(train_data)
    train_data = data_tt.to(torch.float).to(device)
    for i in range(len(other)):
        ticker = tickers1[i]
        _,val,test, _ = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plot)
        data_tt = torch.from_numpy(test)
        test_data[i + n_tickers1] = data_tt.to(torch.float).to(device)
        data_tt = torch.from_numpy(val)
        validation_data[i + n_tickers1] = data_tt.to(torch.float).to(device)

    tickers = np.concatenate((tickers1,other))
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

    f_name = modelsloc +  "vuniversal-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    f_name1 = ticker + "-universal-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"

    PnLs_test = [False] * 10
    PnLs_val = [False] * 10
    means_test = [False] * 10
    means_val = [False] * 10
    print("PnL")
    losstype = "PnL"
    genPnL, discPnL, gen_optPnL, disc_optPnL = TrainLoopMainPnLnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnL.state_dict()}, f_name + "PnL_generator_checkpoint.pth")
    df_temp, PnLs_test[0], PnLs_val[0], means_test[0], means_val[0] = Evaluation3(tickers,freq,genPnL,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.title("Portfolio cummulative PnL " )
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[0]),label=losstype)
    plt.grid(b=True)
    plt.ylabel("bpts")
    plt.legend(loc='best')


    print("PnL MSE")
    genPnLMSE, discPnLMSE, gen_optPnLMSE, disc_optPnLMSE = TrainLoopMainPnLMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSE.state_dict()}, f_name + "PnLMSE_generator_checkpoint.pth")
    df_temp, PnLs_test[1], PnLs_val[1], means_test[1], means_val[1] = Evaluation3(tickers,freq,genPnLMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL MSE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[1]),label=losstype)
    plt.legend(loc='best')

    print("PnL MSE STD")
    genPnLMSESTD, discPnLMSESTD, gen_optPnLMSESTD, disc_optPnLMSESTD = TrainLoopMainPnLMSESTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESTD.state_dict()}, f_name + "PnLMSESTD_generator_checkpoint.pth")
    df_temp, PnLs_test[2], PnLs_val[2], means_test[2], means_val[2]= Evaluation3(tickers,freq,genPnLMSESTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL MSE STD"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[2]),label=losstype)
    plt.legend(loc='best')

    print("PnL MSE SR")
    genPnLMSESR, discPnLMSESR, gen_optPnLMSESR, disc_optPnLMSESR = TrainLoopMainPnLMSESRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESR.state_dict()}, f_name + "PnLMSESR_generator_checkpoint.pth")
    df_temp, PnLs_test[3], PnLs_val[3], means_test[3], means_val[3] = Evaluation3(tickers,freq,genPnLMSESR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL MSE SR"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[3]),label=losstype)
    plt.legend(loc='best')

    print("PnL SR")
    genPnLSR, discPnLSR, gen_optPnLSR, disc_optPnLSR = TrainLoopMainPnLSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSR_generator_checkpoint.pth")
    df_temp, PnLs_test[4], PnLs_val[4], means_test[4], means_val[4] = Evaluation3(tickers,freq,genPnLSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL SR"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[4]),label=losstype)
    plt.legend(loc='best')

    print("PnL STD")
    genPnLSTD, discPnLSTD, gen_optPnLSTD, disc_optPnLSTD = TrainLoopMainPnLSTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSTD_generator_checkpoint.pth")
    df_temp, PnLs_test[5], PnLs_val[5], means_test[5], means_val[5] = Evaluation3(tickers,freq,genPnLSTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL STD"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[5]),label=losstype)
    plt.legend(loc='best')

    print("SR")
    genSR, discSR, gen_optSR, disc_optSR = TrainLoopMainSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "SR_generator_checkpoint.pth")
    df_temp, PnLs_test[6], PnLs_val[6], means_test[6], means_val[6] = Evaluation3(tickers,freq,genSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "SR"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[6]),label=losstype)
    plt.legend(loc='best')

    print("SR MSE")
    genSRMSE, discSRMSE, gen_optSRMSE, disc_optSRMSE = TrainLoopMainSRMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genSRMSE.state_dict()}, f_name + "SRMSE_generator_checkpoint.pth")
    df_temp, PnLs_test[7], PnLs_val[7], means_test[7], means_val[7] = Evaluation3(tickers,freq,genSRMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "SR MSE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[7]),label=losstype)
    plt.legend(loc='best')

    print("MSE")
    genMSE, discMSE, gen_optMSE, disc_optMSE = TrainLoopMainMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genMSE.state_dict()}, f_name + "MSE_generator_checkpoint.pth")
    df_temp, PnLs_test[8], PnLs_val[8], means_test[8], means_val[8] = Evaluation3(tickers,freq,genMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "MSE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[8]),label=losstype)
    plt.legend(loc='best')

    print("ForGAN")
    genFG, discFG, gen_optFG, disc_optFG = TrainLoopForGAN(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genFG.state_dict()}, f_name + "ForGAN_generator_checkpoint.pth")
    df_temp, PnLs_test[9], PnLs_val[9], means_test[9], means_val[9] = Evaluation3(tickers,freq,genFG,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "ForGAN", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "BCE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[9]),label=losstype)
    plt.legend(loc='best')
    plt.savefig(plotsloc+"UniversalPnLCumm.png")

    return results_df, PnLs_test, PnLs_val, means_test, means_val