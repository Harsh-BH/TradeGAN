import FinGAN
import pandas as pd
import matplotlib.pyplot as plt

# Hyperparameters and Configurations
h = 1
l = 10
pred = 1

dataloc = "/home/harsh/Hackathons/TradeGAN/data/"
etflistloc = "/home/harsh/Hackathons/TradeGAN/stocks-etfs-list.csv"

n_epochs = 100
ngpu = 1

loc = "/home/harsh/Hackathons/TradeGAN/Fin-GAN/"
modelsloc = loc + "TrainedModels/"
plotsloc = loc + "Plots/"
resultsloc = loc + "Results/"

tanh_coeff = 100
z_dim = 8
hid_d = 8
hid_g = 8

# Checkpoint and Batch Settings
checkpoint_epoch = 20
batch_size = 100
diter = 1

# Learning Rate Exploration
lrg_s = [0.0001]
lrd_s = [0.0001]
hid_d_s = [8]
hid_g_s = [8]
nres = len(lrg_s)

# Data Split Ratios
tr = 0.8
vl = 0.1
ngrad = 100
vl_later = True

# Plot Configuration
plot = False

# Initial Data Structure for Results
datastart = {'lrd': [], 'lrg': [], 'epochs': [], 'SR_val': []}
results_df = pd.DataFrame(data=datastart)

# Tickers to Analyze
tickers = ['TCS']
corrs = [False] * len(tickers)

# Results Filename
resultsname = "results.csv"
plt.rcParams['figure.figsize'] = [15.75, 9.385]

for j in range(len(hid_d_s)):
    for i in range(nres):
        lrg = lrg_s[i]
        lrd = lrd_s[i]

        for tickern in range(len(tickers)):
            ticker = tickers[tickern]
            print("******************")
            print(f"Processing Ticker: {ticker}")
            print("******************")

            df_temp, corrs[tickern] = FinGAN.FinGAN_combos(
                ticker,
                loc,
                modelsloc,
                plotsloc,
                dataloc,
                etflistloc,
                vl_later,
                lrg,
                lrd,
                n_epochs,
                ngrad,
                h,
                l,
                pred,
                ngpu,
                tanh_coeff,
                tr,
                vl,
                z_dim,
                hid_d,
                hid_g,
                checkpoint_epoch,
                batch_size=batch_size,
                diter=diter,
                plot=plot
            )

            results_df = pd.concat([results_df, df_temp], ignore_index=True)
            results_df.to_csv(resultsloc + resultsname)

            print(f"Completed Processing (FinGAN Combos) for Ticker: {ticker}")

            print("******************")
            print(f"Processing Ticker (LSTM Combos): {ticker}")
            print("******************")

            e = FinGAN.LSTM_combos(
                ticker,
                loc,
                modelsloc,
                plotsloc,
                dataloc,
                etflistloc,
                vl_later=True,
                lrg=0.0001,
                lrd=0.0001,
                n_epochs=500,
                ngrad=100,
                h=1,
                l=10,
                pred=1,
                ngpu=1,
                tanh_coeff=100,
                tr=0.8,
                vl=0.1,
                z_dim=32,
                hid_d=64,
                hid_g=1,
                checkpoint_epoch=20,
                batch_size=100,
                diter=1,
                plot=False,
                freq=2
            )

            print(f"Completed Processing (LSTM Combos) for Ticker: {ticker}")
            print("*************")

print("DONE")