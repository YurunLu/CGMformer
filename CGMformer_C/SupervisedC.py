from multiprocessing import cpu_count
from pathlib import Path
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,auc
from scipy.stats import pearsonr
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR,CosineAnnealingLR
torch.cuda.set_device(0)

note = 'vec_total_823'
embedpath = '../Vec/Shanghai/288/vec_total_823.csv'   #Shanghai/288/vec_total_823.csv  Shanghai/96/97vec_types_823
rawdatapath = '../Data/Shanghai_'
resultspath = '../Results/'
dim = 128
p = 0.4
losseq = 'MSEloss'  #rankingloss  MSEloss
modelnote = losseq

targets = ['age','bmi','fpg','ins0','HOMA-IS','HOMA-B','pg120','hba1c','hdl']

if losseq == 'MSEloss':
    weight = torch.tensor([p]+[(1-p)/len(targets)]*len(targets))    #for MSE loss
else:
    weight = torch.tensor([1/len(targets)]*len(targets))    #for ranking loss
weight = weight.cuda()
input_dim = dim
hidden_dim = [128,64]
output_dim = len(targets)
lr = 0.0001
n_epochs = 10000
mse_loss = torch.nn.MSELoss(reduction='none')
patience = 100

if note not in os.listdir(resultspath):
    os.mkdir(resultspath + note)

def create_datasets(X, y,valid_size=0.2):
    X = torch.from_numpy(X.values)
    y = torch.from_numpy(y.values)
    X = X.to(torch.float32)
    y = y.to(torch.float32)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size)
    print(X_train.shape,y_train.shape)
    X_train, X_valid = [arr.clone().detach() for arr in (X_train, X_valid)]
    y_train, y_valid = [arr.clone().detach() for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return train_ds, valid_ds

def create_loaders(train_ds, valid_ds, bs=128, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hiddendim, outputdim):
        super(MLP, self).__init__()
        layers = []
        indim = input_dim
        for dim in hiddendim+[1]:
            layers.append(torch.nn.Linear(indim, dim))
            layers.append(torch.nn.Sigmoid())
            indim = dim
        self.encoder = torch.nn.ModuleList(layers)
        self.decoder = torch.nn.Linear(1, outputdim)
        self.decoder_act = torch.nn.Sigmoid()
    def forward(self, x):
        for l in self.encoder:
            x = l(x)
        out = self.decoder(x)
        out = self.decoder_act(out)
        x = torch.cat((x, out), 1)
        return x


class RankingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(RankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, targets):
        # Compute the ranking loss for each sample in the batch
        loss_per_sample = (scores - scores.t())

        # Create masks for different target values
        mask_0 = targets == 0
        mask_5 = targets == 0.5
        mask_1 = targets == 1

        # Sum the ranking loss for each sample, based on the masks
        m_0_vs_05 = loss_per_sample[mask_5.view(-1)][:,mask_0.view(-1)]
        m_05_vs_1 = loss_per_sample[mask_1.view(-1)][:,mask_5.view(-1)]

        #loss_0_vs_05 = -torch.mean(torch.min(torch.zeros(m_0_vs_05.shape).cuda(), m_0_vs_05))
        #loss_05_vs_1 = -torch.mean(torch.min(torch.zeros(m_05_vs_1.shape).cuda(),m_05_vs_1))
        #loss_0_vs_05 = torch.sum(m_0_vs_05 < 0) / (m_0_vs_05.shape[0] * m_0_vs_05.shape[1])
        #loss_05_vs_1 = torch.sum(m_05_vs_1 < 0) / (m_05_vs_1.shape[0] * m_05_vs_1.shape[1])

        loss_0_vs_05 = 1 - torch.mean(m_0_vs_05)
        loss_05_vs_1 = 1 - torch.mean(m_05_vs_1)

        # Sum the losses for the entire batch
        rank_loss = loss_0_vs_05 + loss_05_vs_1

        return rank_loss

def maskedMSE(prey,truey):
    mask = (~truey.isnan()).clone()
    mask = mask.float()
    truey = torch.where(torch.isnan(truey), torch.full_like(truey, 0), truey)
    return torch.sum(mask * mse_loss(prey.float(), truey.float()) * weight) / mask.sum()

def getdata(embedpath=embedpath,rawdatapath=rawdatapath,dim=dim,scaleHOMA=True):
    vec = pd.read_csv(embedpath, index_col=0)
    label = pd.read_csv(rawdatapath + 'Label.csv', index_col=0)
    vec = vec.sort_values('index')
    vec.index = vec['index']
    vec = vec.iloc[:, :dim]
    label = label.loc[vec.index]
    if scaleHOMA and ('HOMA-B' in label.columns):
        label['HOMA-B'] = label['HOMA-B'].where((0 <= label['HOMA-B']) & (label['HOMA-B'] <= 400))
        label['HOMA-IS'] = label['HOMA-IS'].where((1/5 <= label['HOMA-IS']) & (label['HOMA-IS'] <= 5))
        ids = label[pd.notna(label['HOMA-IS'])].index
        label.loc[ids, 'HOMA-IR'] = [1 / x for x in label.loc[ids, 'HOMA-IS']]
    print(f'Import data {len(vec)} sample dim={dim}')
    return label, vec

if __name__=='__main__':
    print('Preparing datasets')
    label, vec = getdata(embedpath, rawdatapath, dim)
    Type2course = {'NGT':0,'IGR':0.5,'T2D':1}
    label['Course'] = [Type2course[t] for t in label['type']]
    value = label[['Course']+targets]
    value = (value-value.min(axis=0))/(value.max(axis=0)-value.min(axis=0))
    print(vec.shape,value.shape)
    train_ds, valid_ds= create_datasets(vec, value)
    bs = 64
    N_train = len(train_ds)
    print(N_train)
    print(f'Creating data loaders with batch size: {bs}')
    trn_dl, val_dl = create_loaders(train_ds, valid_ds, bs)  # , jobs=cpu_count()
    iterations_per_epoch = len(trn_dl)
    best_mse = 10000
    all_loss_train = []
    all_loss_val = []
    model = MLP(dim, hidden_dim, output_dim)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = ExponentialLR(opt, gamma=0.1)
    print('Start model training')
    trials = 0
    LabelRankingLoss = RankingLoss(margin=1.0)
    for epoch in range(1, n_epochs + 1):
        loss_epoch = 0
        for i, (x_batch, y_batch) in enumerate(trn_dl):
            model.train()
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            #print(y_batch.shape)
            opt.zero_grad()
            out = model(x_batch)
            if losseq == 'MSEloss':
                loss = maskedMSE(out, y_batch)
            else:
                loss = (1-p)*maskedMSE(out[:,1:],y_batch[:,1:])+p*LabelRankingLoss(out[:,0:1],y_batch[:,0:1])
            loss_epoch += loss.item()*x_batch.shape[0]
            loss.backward()
            opt.step()
            #sched.step()
        all_loss_train.append(loss_epoch/N_train)
        model.eval()
        mse_val = 0
        with torch.no_grad():
            val_out = []
            val_all = []
            for x_val, y_val in val_dl:
                x_val = x_val.cuda()
                y_val = y_val.cuda()
                val_out.append(model(x_val).clone())
                val_all.append(y_val.clone())
            out = torch.cat(val_out,0)
            target = torch.cat(val_all,0)
            if losseq == 'MSEloss':
                mse_val = maskedMSE(out, target).item()
            else:
                mse_val = ((1-p)*maskedMSE(out[:,1:],target[:,1:])+p*LabelRankingLoss(out[:,0:1],target[:,0:1])).item()
        all_loss_val.append(mse_val)
        if mse_val < best_mse:
            trials = 0
            best_mse = mse_val
            torch.save(model.state_dict(), resultspath + note+'/%s_BestModelForCS.pth'%modelnote)
            print(f'Epoch {epoch} best model saved with loss: {mse_val:.4e}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
        #print(f'Epoch {epoch} train MSE: {all_loss_train[-1]:.4f}, val MSE: {all_loss_val[-1]:.4f}')
    plt.plot(range(len(all_loss_train)), all_loss_train,label='Train Loss')
    plt.plot(range(len(all_loss_val)), all_loss_val, label='Valid Loss')
    plt.legend()
    plt.savefig(resultspath + note+'/%s_Loss.png'%modelnote)