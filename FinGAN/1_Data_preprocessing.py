import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rnd

def ETF_find(etflistloc, stock):
    """
    reading a file containing information on stock memberships
    input: stock ticker
    output: corresponding ETF ticker
    """
    data = pd.read_csv(etflistloc)
    out = np.array(data['ticker_y'][data['ticker_x']==stock])[0]
    return out

def excessreturns_closeonly(dataloc, stock, etf, plotcheck = False):
    """
    function to get a time series of DAILY CLOSING
    etf-excess log returns for a given stock
    all prices are adjusted for stock events
    input: location of datasets, stock ticker, etf ticker
    output: time series of etf excess log returns
    optional: plot sanity check
    """
    s_df = pd.read_csv(dataloc+stock+".csv")
    e_df = pd.read_csv(dataloc+etf+".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = (dates_dt < d1)
    s_df = s_df[smp]
    e_df = e_df[smp]
    s_log = np.log(s_df['AdjClose'])
    e_log = np.log(e_df['AdjClose'])
    dates_dt = dates_dt[smp]
    s_ret = np.diff(s_log)
    e_ret = np.diff(e_log)
    excessret = s_ret - e_ret

    if plotcheck:
        plt.figure(stock+" price")
        plt.title(stock+" price")
        plt.plot(dates_dt,s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns "+stock)
        plt.title("Returns "+stock)
        plt.plot(dates_dt[1:],s_ret, alpha = 0.7, label = 'stock')
        plt.plot(dates_dt[1:],e_ret, alpha = 0.7, label = 'etf')
        plt.plot(dates_dt[1:],excessret, alpha = 0.7, label = 'excess return')
        plt.xlabel("date")
        plt.legend()
        plt.show()
    return excessret, dates_dt[1:]

def excessreturns(dataloc, stock, etf, plotcheck = False):
    """
    function to get a time series of alternating close and open
    etf-excess log returns for a given stock
    all prices are adjusted for stock events
    input: location of datasets, stock ticker, etf ticker
    output: time series of etf excess log returns
    optional: plot sanity check
    """
    s_df = pd.read_csv(dataloc+stock+".csv")
    e_df = pd.read_csv(dataloc+etf+".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = (dates_dt < d1)
    s_df = s_df[smp]
    dates_dt = pd.to_datetime(s_df['date'])
    e_df = e_df[smp]
    s_logclose = np.log(s_df['AdjClose'])
    e_logclose = np.log(e_df['AdjClose'])
    s_logopen = np.log(s_df['AdjOpen'])
    e_logopen = np.log(e_df['AdjOpen'])
    s_log = np.zeros(2*len(s_logclose))
    e_log = np.zeros(2*len(s_logclose))
    for i in range(len(s_logclose)):
        s_log[2 * i] = s_logopen[i]
        s_log[2 * i + 1] = s_logclose[i]
        e_log[2 * i] = e_logopen[i]
        e_log[2 * i + 1] = e_logclose[i]
    s_ret = np.diff(s_log)
    e_ret = np.diff(e_log)
    s_ret[s_ret > 0.15] = 0.15
    s_ret[s_ret < -0.15] = -0.15
    e_ret[e_ret > 0.15] = 0.15
    e_ret[e_ret < -0.15] = -0.15
    excessret = s_ret - e_ret
    dates_dt = pd.to_datetime(s_df['date'])
    if plotcheck:
        plt.figure(stock+" price")
        plt.title(stock+" price")
        plt.plot(dates_dt,s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns "+stock)
        plt.title("Returns "+stock)
        plt.plot(range(len(s_ret)),s_ret, alpha = 0.7, label = 'stock')
        plt.plot(range(len(e_ret)),e_ret, alpha = 0.7, label = 'etf')
        plt.plot(range(len(e_ret)),excessret, alpha = 0.7, label = 'excess return')
        plt.legend()
        plt.show()
    return excessret, dates_dt

def rawreturns(dataloc, stock, plotcheck = False):
    """
    function to get a time series of raw log returns for a given stock/etf
    all prices are adjusted for stock events
    input: location of datasets, stock ticker, etf ticker
    output: time series of etf excess log returns
    optional: plot sanity check
    """
    s_df = pd.read_csv(dataloc+stock+".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = (dates_dt < d1)
    s_df = s_df[smp]
    dates_dt = pd.to_datetime(s_df['date'])
    s_logclose = np.log(s_df['AdjClose'])
    s_logopen = np.log(s_df['AdjOpen'])
    s_log = np.zeros(2*len(s_logclose))
    for i in range(len(s_logclose)):
        s_log[2 * i] = s_logopen[i]
        s_log[2 * i + 1] = s_logclose[i]
    s_ret = np.diff(s_log)
    s_ret[s_ret > 0.15] = 0.15
    s_ret[s_ret < -0.15] = -0.15
    dates_dt = pd.to_datetime(s_df['date'])
    if plotcheck:
        plt.figure(stock+" price")
        plt.title(stock+" price")
        plt.plot(dates_dt,s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns "+stock)
        plt.title("Returns "+stock)
        plt.plot(range(len(s_ret)),s_ret)
        plt.legend()
        plt.show()
    return s_ret, dates_dt

def split_train_val_test(stock, dataloc, etflistloc, tr = 0.8, vl = 0.1, h = 1, l = 10, pred = 1, plotcheck=False):
    """
    prepare etf excess log returns for a given stock
    split into train, val, test
    h: sliding window
    l: condition window (number of previous values)
    pred: prediction window
    """
    etf = ETF_find(etflistloc, stock)
    excess_returns, dates_dt = excessreturns(dataloc, stock, etf, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr*N)
    N_vl = int(vl*N)
    N_tst = N - N_tr - N_vl
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    test_sr = excess_returns[N_tr+N_vl:]
    n = int((N_tr-l-pred)/h)+1
    train_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i,:] = train_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_vl-l-pred)/h)+1
    val_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        val_data[i,:] = val_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_tst-l-pred)/h)+1
    test_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i,:] = test_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    if plotcheck:
        plt.figure("Excess returns")
        plt.plot(dates_dt,excess_returns)
        plt.title(stock+ " excess returns")
        plt.axvline(x = dates_dt[N_tr],color = "red")
        plt.axvline(x = dates_dt[N_tr+N_vl],color = "red")
        plt.show()
    return train_data,val_data,test_data, dates_dt

def split_train_testraw(stock, dataloc, tr = 0.8, vl = 0.1, h = 1, l = 10, pred = 1, plotcheck=False):
    """
    prepare raw log returns for a given stock
    split into train, test
    h: sliding window
    l: condition window (number of previous values)
    pred: prediction window
    """
    excess_returns, dates_dt = rawreturns(dataloc, stock, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr*N) + int(vl*N)
    N_tst = N - N_tr
    train_sr = excess_returns[0:N_tr]
    test_sr = excess_returns[N_tr:]
    n = int((N_tr-l-pred)/h)+1
    train_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i,:] = train_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_tst-l-pred)/h)+1
    test_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i,:] = test_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h

    return train_data,test_data

def split_train_val_testraw(stock, dataloc, tr = 0.8, vl = 0.1, h = 1, l = 10, pred = 1, plotcheck=False):
    """
    prepare raw log returns for a given stock
    split into train, val, test
    h: sliding window
    l: condition window (number of previous values)
    pred: prediction window
    """
    excess_returns, dates_dt = rawreturns(dataloc, stock, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr*N)
    N_vl = int(vl*N)
    N_tst = N - N_tr - N_vl
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    test_sr = excess_returns[N_tr+N_vl:]
    n = int((N_tr-l-pred)/h)+1
    train_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i,:] = train_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_vl-l-pred)/h)+1
    val_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        val_data[i,:] = val_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_tst-l-pred)/h)+1
    test_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i,:] = test_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    if plotcheck:
        plt.figure("returns")
        plt.plot(dates_dt,excess_returns)
        plt.title(stock+ " =returns")
        plt.axvline(x = dates_dt[N_tr],color = "red")
        plt.axvline(x = dates_dt[N_tr+N_vl],color = "red")
        plt.show()
    return train_data,val_data,test_data, dates_dt