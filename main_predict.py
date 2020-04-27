import argparse
import json
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from modules import Encoder, Decoder
from utils import numpy_to_tvar
import utils

parser = argparse.ArgumentParser(description='Dual Stage Attention Model for Time Series prediction')
parser.add_argument('--data', type=str, default='train.csv',
                    help='location/name of the datafile')
# parser.add_argument('--epochs', type=int, default=50000,
#                     help='number of epochs')
# parser.add_argument('--T', type=int, default=5,
#                     help='time steps')
# parser.add_argument('--batch_size', type=int, default=128,
#                     help='batch_size')
parser.add_argument('--gpu', type=int, default=3, help='GPU device to use')
parser.add_argument('--seed', type=int, default=12,
                    help='random seed')
parser.add_argument('--save', type=str,
                    help='directory where model is saved', required=True)
args = parser.parse_args()


class TrainData:
    def __init__(self, feats, targs):
        self.feats = feats
        self.targs = targs

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

def preprocess_data(dat, col_names, scaleX, scaley):
    # proc_dat = scale.transform(dat)

    # mask = np.ones(proc_dat.shape[1], dtype=bool)
    # dat_cols = list(dat.columns)
    # for col_name in col_names:
    #     mask[dat_cols.index(col_name)] = False

    # feats = proc_dat[:, mask]
    # targs = proc_dat[:, ~mask]

    y_dat = dat[list(col_names)]
    X_dat = dat[[names for names in dat.columns if names not in col_names]]

    feats1 = scaleX.transform(X_dat)
    targs1 = scaley.transform(y_dat)
    # feats1 = X_dat.to_numpy()
    # targs1 = y_dat.to_numpy() - np.mean(y_dat.to_numpy())
    # print("mean y:", np.mean(y_dat.to_numpy()))

    return TrainData(feats1, targs1)


def predict(encoder, decoder, t_dat, batch_size, T):
    y_pred = np.zeros((t_dat.feats.shape[0] - T + 1, t_dat.targs.shape[1]))
    print("feats: ", t_dat.feats.shape)
    print("targs: ", t_dat.targs.shape)
    print("y_pred:", y_pred.shape, len(y_pred))
    print("batch_size:",batch_size)
    print("T:",T)
    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        print(batch_idx)
        b_len = len(batch_idx)
        print("b_len:",b_len)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        print("X:",X.shape)
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))
        print("y_history:",y_history.shape)

        for b_i, b_idx in enumerate(batch_idx):
            # print("b_i: ",b_i,"b_idx:",b_idx)
            idx = range(b_idx, b_idx + T - 1)
            # print("feats:",t_dat.feats[idx, :])
            # print("targs:",t_dat.targs[idx])
            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        print("y_history:",y_history.shape)
        _, input_encoded = encoder(numpy_to_tvar(X))
        y_pred[y_slc] = decoder(input_encoded, y_history).cpu().data.numpy()
    print("y_pred:",y_pred.shape)
    return y_pred


debug = False
save_plots = False

enc_params = pd.read_csv(os.path.join('results',args.save,'enc_params.csv'))
dec_params = pd.read_csv(os.path.join('results',args.save,'dec_params.csv'))
print(enc_params)
print(dec_params)
enc = Encoder(input_size = enc_params['input_size'][0].item(), hidden_size = enc_params['hidden_size'][0].item(), T = enc_params['T'][0].item()).cuda()
enc.load_state_dict(torch.load(os.path.join("results", args.save, "encoder.pt"), map_location=device))

dec = Decoder(encoder_hidden_size = dec_params['encoder_hidden_size'][0].item(), decoder_hidden_size = dec_params['decoder_hidden_size'][0].item(), T = dec_params['T'][0].item(), out_feats = dec_params['out_feats'][0].item()).cuda()
dec.load_state_dict(torch.load(os.path.join("results", args.save, "decoder.pt"), map_location=device))

scaleX = joblib.load(os.path.join("results",args.save, "scaleX.pkl"))
scaley = joblib.load(os.path.join("results",args.save, "scaley.pkl"))

raw_data = pd.read_csv(os.path.join('data','covid',args.data))

del raw_data['Id']
del raw_data['Date']
del raw_data['Province_State']
del raw_data['Country_Region']
print(raw_data[140*90:141*90].head())
targ_cols = ('Fatalities',)
data = preprocess_data(raw_data[140*90:141*90], targ_cols, scaleX, scaley)

da_rnn_params = pd.read_csv(os.path.join('results',args.save,'da_rnn_params.csv'))
print(da_rnn_params)
final_y_pred = predict(enc, dec, data, batch_size=da_rnn_params['batch_size'][0].item(), T=da_rnn_params['T'][0].item())

print('final_y_pred: ', type(final_y_pred))
print('final_y_pred: ', (scaley.inverse_transform(final_y_pred)).astype(int))
# plt.figure()
# plt.plot(final_y_pred, label='Predicted')
# plt.plot(data.targs[(da_rnn_kwargs["T"]-1):], label="True")
# plt.legend(loc='upper left')
# utils.save_or_show_plot("final_predicted_reloaded.png", save_plots)
