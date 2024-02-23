from multiprocessing import cpu_count
from pathlib import Path
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR,CosineAnnealingLR
from SampleUnsupervised import getdata,drawSingle
import SupervisedC
torch.cuda.set_device(0)  # if you have more than one CUDA device

'''
note = 'zhao97_vec_823'
embedpath = '../Vec/zhao/zhao97_vec_823.csv'
#Shanghai/288/vec_total_823.csv  Shanghai/96/97vec_types_823  Colas_emb/Colas_vec.csv zhao/zhao97_vec_823
rawdatapath = '../Data/Zhao_'
VecKey = 'DayLevel'
'''

note = 'zhao_vec_merge'
embedpath = '../Vec/zhao/zhao_vec_merge.csv'
rawdatapath = '../../Data/Summary_Zhao.csv'
VecKey = 'SampleLevel'

note = 'vec_total_823_mergevec'
embedpath = '../Vec/Shanghai/288/vec_total_823_mergevec.csv'
rawdatapath = '../../Data/Summary_Shanghai.csv'
VecKey = 'SampleLevel'

resultspath = '../Results/'
modelpath = resultspath+'vec_total_823'
#vec_total_823  97vec_types_823
modelnote = 'MSEloss'  #rankingloss  MSEloss
dim = 128

targets = ['age','bmi','fpg','ins0','HOMA-IS','HOMA-B','pg120','hba1c','hdl']

input_dim = dim
hidden_dim = [128,64]
output_dim = len(targets)


if note not in os.listdir(resultspath):
    os.mkdir(resultspath + note)

def create_datasets(X, y):
    X = torch.from_numpy(X.values).to(torch.float32)
    y = torch.from_numpy(y.values).to(torch.float32)
    ds = TensorDataset(X, y)
    return ds

def getsamplevec(embedpath, rawdatapath):
    vec = pd.read_csv(embedpath, index_col=0)
    label = pd.read_csv(rawdatapath)
    label.index = label['id']
    label = label.loc[vec.index]
    return label,vec

if __name__=='__main__':
    print('Preparing datasets')
    if VecKey == 'DayLevel':
        label, vec = getdata(embedpath, rawdatapath, dim,scaleHOMA=True)
    else:
        label, vec = getsamplevec(embedpath, rawdatapath)

    Type2course = {'NGT':0,'IGR':0.5,'T2D':1,'T1D':1}
    label[list(set(targets)-set(label.columns))]=0
    label['Course'] = [Type2course[t.split('_')[1]] for t in label['id']]

    value = label[['Course']+targets]
    value = (value-value.min(axis=0))/(value.max(axis=0)-value.min(axis=0))
    print(vec.shape,value.shape)
    ds = create_datasets(vec, value)
    bs = 64
    print(f'Creating data loaders with batch size: {bs}')
    dl = DataLoader(ds, bs, shuffle=False)
    print('Calculate Supervised Course of Disease')
    model = SupervisedC.MLP(dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(modelpath+'/%s_BestModelForCS.pth'%modelnote))
    model.eval()
    model = model.cuda()
    model.eval()
    mse_val = 0
    with torch.no_grad():
        val_out = []
        val_all = []
        for x_val, y_val in dl:
            x_val = x_val.cuda()
            y_val = y_val.cuda()
            val_out.append(model(x_val).clone())
            val_all.append(y_val.clone())
    val_out = torch.cat(val_out,0).cpu()
    print(val_out.shape)
    #drawtargets = ['age','bmi','hba1c','tc','ldl','hdl','fpg','ins0','cp0','pg120','duration']
    drawtargets = ['age','bmi','hba1c','tc','ldl','hdl','fpg','ins0','cp0','pg120','duration','HOMA-IS','HOMA-B']
    label['C'] = val_out[:,0]
    label.to_csv(resultspath + note+'/%s_CS.csv'%modelnote)
    label.index = range(len(label))

    drawSingle(val_out, label, resultspath + note, drawtargets, modelnote + '_Cs')
