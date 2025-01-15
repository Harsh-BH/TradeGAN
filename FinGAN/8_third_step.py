def LSTM_combos(ticker,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later = True, lrg = 0.0001, lrd = 0.0001, n_epochs = 500, ngrad = 100, h = 1, l = 10, pred = 1, ngpu = 1, tanh_coeff = 100, tr = 0.8, vl = 0.1, z_dim = 32, hid_d = 64, hid_g = 8, checkpoint_epoch = 20, batch_size = 100, diter = 1, plot = False, freq = 2):
    """
    Training and evaluation on (test and val) of LSTM and LSTM-Fin
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

    condition_size = l
    target_size = pred
    ref_mean = torch.mean(train_data[0:batch_size,:])
    ref_std = torch.std(train_data[0:batch_size,:])

    gen = LSTM(noise_dim = 0,cond_dim=condition_size, hidden_dim=hid_g,output_dim=pred,mean =ref_mean,std=ref_std)
    gen.to(device)
    criterion = False

    PnL_test = [False] * 6
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)

    gen, gen_opt,  alpha, beta, gamma, delta = GradientCheckLSTM(ticker, gen, gen_opt, ngrad, train_data,batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)

    f_name = modelsloc + ticker + "-LSTM-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    f_name1 = ticker + "-LSTM-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"

    print("PnL")
    losstype = "PnL"
    genPnL,  gen_optPnL = TrainLoopnLSTMPnL(gen,  gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnL.state_dict()}, f_name + "PnL_lstm_checkpoint.pth")
    df_temp, PnL_test[0], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnL,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)

    ntest = test_data.shape[0]
    pd.DataFrame(PnL_test[0]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
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

    print("PnL STD")
    losstype = "PnL STD"
    genPnLMSESTD, gen_optPnLMSESTD = TrainLoopnLSTMPnLSTD(gen,  gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESTD.state_dict()}, f_name + "PnLMSESTD_lstm_checkpoint.pth")
    df_temp, PnL_test[1], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnLMSESTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL STD LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[1]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")

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
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')

    print("PnL SR")
    losstype = "PnL SR"
    genPnLMSESR, gen_optPnLMSESR = TrainLoopnLSTMPnLSR(gen, gen_opt,  criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESR.state_dict()}, f_name + "PnLMSESR_lstm_checkpoint.pth")
    df_temp, PnL_test[2], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnLMSESR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL SR LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[2]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")

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
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')


    print("STD")
    losstype = "STD"
    genPnLSR, gen_optPnLMSESR = TrainLoopnLSTMSTD(gen, gen_opt,  criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "STD_lstm_checkpoint.pth")
    df_temp, PnL_test[3], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnLSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "STD LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[3]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")

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
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')


    print("SR")
    losstype = "SR"
    genSR, gen_optSR = TrainLoopnLSTMSR(gen,  gen_opt , criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "SR_lstm_checkpoint.pth")
    df_temp, PnL_test[4], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[4]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
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
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')


    print("MSE")
    losstype = "MSE"
    genMSE, gen_optMSE = TrainLoopnLSTM(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genMSE.state_dict()}, f_name + "MSE_lstm_checkpoint.pth")
    df_temp, PnL_test[5], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[5]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')
    plt.savefig(plotsloc+"LSTM-"+ticker+"-cummulativePnL.png")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-intradayPnL.png")

        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-overnight.png")
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-overnight.png")

        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-intradayPnL.png")

    corrm = np.corrcoef(PnL_test)

    return results_df, corrm