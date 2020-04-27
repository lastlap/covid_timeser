import argparse
import collections
import json
import os, sys, glob
import logging
import time

import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
from modules import Encoder, Decoder
from utils import numpy_to_tvar
from utils import RMSLELoss
rmsleloss = RMSLELoss()

parser = argparse.ArgumentParser(description='Dual Stage Attention Model for Time Series prediction')
parser.add_argument('--data', type=str, default='train.csv',
                    help='location/name of the datafile')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs')
parser.add_argument('--T', type=int, default=5,
                    help='time steps')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch_size')
parser.add_argument('--lr', type=int, default=0.01,
                    help='learning rate')
parser.add_argument('--min_lr', type=int, default=0.0001,
                    help='learning rate')
parser.add_argument('--clip', type=float, default=0.1,
                    help='gradient clipping')
parser.add_argument('--gpu', type=int, default=3, help='GPU device to use')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--seed', type=int, default=12,
                    help='random seed')
args = parser.parse_args()

# if torch.cuda.is_available():
torch.cuda.set_device(args.gpu)
    # cudnn.benchmark = True
    # cudnn.enabled=True
    # torch.cuda.manual_seed_all(args.seed)

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

save_name = 'main-{}-{}'.format('EXP', time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('results',save_name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info('Args: {}'.format(args))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# logging.info(f"Using computation device: {device}")



class TrainConfig:
    def __init__(self, T, train_size, batch_size, loss_func):
        self.T = T
        self.train_size = train_size
        self.batch_size = batch_size
        self.loss_func = loss_func

class TrainData:
    def __init__(self, feats, targs):
        self.feats = feats
        self.targs = targs

DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt", "enc_sch", "dec_sch"])

def preprocess_data(dat, col_names):
    # scale = StandardScaler().fit(dat)
    # proc_dat = scale.transform(dat)

    # mask = np.ones(proc_dat.shape[1], dtype=bool)
    # dat_cols = list(dat.columns)
    # for col_name in col_names:
    #     mask[dat_cols.index(col_name)] = False

    # feats = proc_dat[:, mask]
    # targs = proc_dat[:, ~mask]

    y_dat = dat[list(col_names)]
    X_dat = dat[[names for names in dat.columns if names not in col_names]]

    scaleX = StandardScaler().fit(X_dat)
    scaley = StandardScaler().fit(y_dat)
    feats1 = scaleX.transform(X_dat)
    targs1 = scaley.transform(y_dat)
    # feats1 = X_dat.to_numpy()
    # targs1 = y_dat.to_numpy() - np.mean(y_dat.to_numpy())
    # print("mean y:", np.mean(y_dat.to_numpy()))

    return TrainData(feats1, targs1), scaleX, scaley


def da_rnn(train_data, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):

    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logging.info(f"Training size: {train_cfg.train_size:d}.")

    enc_params = pd.DataFrame([{'input_size': train_data.feats.shape[1], 'hidden_size': encoder_hidden_size, 'T': T}])
    enc_params.to_csv(os.path.join('results', save_name, 'enc_params.csv'))
    
    encoder = Encoder(input_size = enc_params['input_size'][0].item(), hidden_size = enc_params['hidden_size'][0].item(), T = enc_params['T'][0].item()).cuda()

    dec_params = pd.DataFrame([{'encoder_hidden_size': encoder_hidden_size,
                  'decoder_hidden_size': decoder_hidden_size, 'T': T, 'out_feats': n_targs}])
    dec_params.to_csv(os.path.join('results', save_name, 'dec_params.csv'))
    
    decoder = Decoder(encoder_hidden_size = dec_params['encoder_hidden_size'][0].item(), decoder_hidden_size = dec_params['decoder_hidden_size'][0].item(), T = dec_params['T'][0].item(), out_feats = dec_params['out_feats'][0].item()).cuda()
    
    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=args.wdecay)
    
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=args.wdecay)
    
    encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, train_data.feats.shape[0], eta_min = args.min_lr)
    decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, train_data.feats.shape[0], eta_min = args.min_lr)

    model = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler)

    return train_cfg, model


def train(model, train_data, t_cfg, n_epochs=10, save_plots=False):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logging.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0
    stored_loss = 100000000

    for e_i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T) # train_size - T is done so that when we take the next T samples for training we don't get an index error

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            # print("batch_idx: ",len(batch_idx))
            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)

            loss = train_iteration(model, t_cfg.loss_func, feats, y_history, y_target)
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            n_iter += 1

            # adjust_learning_rate(model, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])
        
        # if e_i % 500 ==0:
        #     model = model._replace(enc_sch = optim.lr_scheduler.CosineAnnealingLR(model.enc_opt, args.epochs, eta_min = args.min_lr))
        #     model = model._replace(dec_sch = optim.lr_scheduler.CosineAnnealingLR(model.dec_opt, args.epochs, eta_min = args.min_lr))       

        if e_i % 10 == 0 or e_i == n_epochs - 1:
            y_test_pred = predict(model, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
            rmse = (torch.sqrt(t_cfg.loss_func(numpy_to_tvar(y_test_pred), numpy_to_tvar(train_data.targs[t_cfg.train_size:])))).cpu().data.numpy()
            rmsle = (rmsleloss(numpy_to_tvar(y_test_pred), numpy_to_tvar(train_data.targs[t_cfg.train_size:]))).cpu().data.numpy()
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
            logging.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}, rmse loss: {rmse}, rmsle loss: {rmsle}")

            if rmsle < stored_loss:
                stored_loss = rmsle
                torch.save(model.encoder.state_dict(), os.path.join("results",save_name, "encoder.pt"))
                torch.save(model.decoder.state_dict(), os.path.join("results",save_name, "decoder.pt"))
                logging.info("Saving model")

    return iter_losses, epoch_losses


def prep_train_data(batch_idx, t_cfg, train_data):
    # len(batch_idx) = batch size =128 here
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    # feats consists of len(batch_idx) number of matrices where each matrix has (T-1)*train_data.feats.shape[1] because for each batch index in batch_idx we have T-1 time steps with train_data.feats.shape[1] features
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    # y_history contains len(batch_idx) number of matrices of dimension (T-1)*train_data.targs.shape[1]
    # here y_history is just T-1 time steps of previous values - 128 such arrays.
    y_target = train_data.targs[batch_idx + t_cfg.T]
    # y_target contains len(batch_idx) number of results
    # print("y_target: ",y_target.shape)
    # print("y_history: ",y_history.shape)
    # print("feats: ",feats.shape)
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :] # 1st matrix = 1st (T-1) time steps starting from 1st batch index in batch_idx and so on
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(model, n_iter):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(model.enc_opt.param_groups, model.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(model, loss_func, X, y_history, y_target):
    model.enc_opt.zero_grad()
    model.dec_opt.zero_grad()

    input_weighted, input_encoded = model.encoder(numpy_to_tvar(X))
    y_pred = model.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    # print("loss_func y_pred:",y_pred.shape,type(y_pred))
    # print("loss_func y_true:",y_true.shape,type(y_true))
    loss = torch.sqrt(loss_func(y_pred, y_true))
    loss.backward()
    
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]

    torch.nn.utils.clip_grad_norm_(encoder_params, args.clip)
    torch.nn.utils.clip_grad_norm_(decoder_params, args.clip)

    model.enc_opt.step()
    model.dec_opt.step()

    model.enc_sch.step()
    model.dec_sch.step()

    return loss.item()


def predict(model, t_dat, train_size, batch_size, T, on_train=False):
    out_size = t_dat.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = model.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = model.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred


save_plots = True
debug = False

raw_data = pd.read_csv(os.path.join('data','covid', args.data))
# raw_data.dropna(how='any',inplace=True)

countries_by_id = pd.DataFrame(columns=['Id','Country','Province'])

for i in range(0,raw_data.shape[0],90):
    countries_by_id = countries_by_id.append({'Id':i//90, 'Country':raw_data['Country_Region'][i], 'Province':raw_data['Province_State'][i]},ignore_index=True)
    # print(raw_data.loc[i,'Country_Region'],raw_data.loc[i,'Province_State'])

countries_by_id.to_csv(os.path.join('results', save_name, 'countries_by_id.csv'),index=None,header=True)

del raw_data['Id']
del raw_data['Date']
del raw_data['Province_State']
del raw_data['Country_Region']

logging.info(f"Shape of data: {raw_data[140*90:141*90].shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
targ_cols = ('Fatalities',)
data, scaleX, scaley = preprocess_data(raw_data[140*90:141*90], targ_cols)

da_rnn_params = pd.DataFrame([{"batch_size": args.batch_size, "T": args.T}])
config, model = da_rnn(data, n_targs=len(targ_cols), learning_rate=args.lr, batch_size = da_rnn_params['batch_size'][0], T = da_rnn_params['T'][0])

iter_loss, epoch_loss = train(model, data, config, n_epochs=args.epochs, save_plots=save_plots)

final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)
da_rnn_params.to_csv(os.path.join('results', save_name, 'da_rnn_params.csv'))
joblib.dump(scaleX, os.path.join("results",save_name, "scaleX.pkl"))
joblib.dump(scaley, os.path.join("results",save_name, "scaley.pkl"))