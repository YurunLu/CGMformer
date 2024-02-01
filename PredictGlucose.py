import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale,MinMaxScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import PredictMetrics

bs = 64
latentdim = 32
layer_dim = 3
lr = 0.0001
n_epochs = 5000
modelpath = 'PredictGlucose Alluse'

NeedTrain = True

CVtime = 5
resultfile = './Glucose_result_Predict.csv'

Perturb = None   #Raw RawHeat HighHeat None Standard
Fiber = ''
Ratio = (4,2,4)

recordresult = False
NeedScatter = False
if Perturb:
    if Perturb == 'Raw':
        resultfile = './Glucose_result_Perturb_Raw.csv'
    elif Perturb=='HighHeat':
        resultfile = './Glucose_result_Perturb_HighHeat.csv'
    elif Perturb == 'Standard':
        resultfile = './Glucose_result_Perturb_Standard.csv'
    elif Perturb=='RawHeat':
        resultfile = './Glucose_result_Perturb_' + ''.join([str(x) for x in Ratio]) + Fiber+'.csv'
    CVtime = 1
    recordresult = True

class GlucosePredict(nn.Module):
    def __init__(self, hidden_dim, layer_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.emb_encode = nn.Linear(128,hidden_dim)
        self.rnn = nn.LSTM(6, hidden_dim, layer_dim, batch_first=True)
        self.decode = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = x[:, :128]
        dietary = x[:, 128:128+5]
        G = x[:, 128 + 5:]

        emb = self.emb_encode(emb)
        #dietary = torch.zeros(dietary.size(0),4,5).cuda()  #if without meal
        dietary = torch.cat((torch.zeros(G.size(0), 3, 5).cuda(), torch.unsqueeze(dietary, 1)), 1)

        h0, c0 = torch.zeros(self.layer_dim, G.size(0), self.hidden_dim).cuda(), torch.zeros(self.layer_dim, G.size(0), self.hidden_dim).cuda()
        h0 = emb.unsqueeze(0).repeat(self.layer_dim, 1, 1)   #mask if without patient info

        G = torch.unsqueeze(G, 2)
        G = torch.cat((G, dietary), 2)

        out, (hn, cn) = self.rnn(G, (h0, c0))
        ExtG = self.decode(out)[:, :-1, 0]
        predG = []
        for i in range(8):
            predG.append(self.decode(out[:, -1, :]))
            inputnew = torch.unsqueeze(predG[-1], 1)
            inputnew = torch.cat((inputnew, torch.zeros(inputnew.shape[0], 1, 5).cuda()), 2)
            out, (hn, cn) = self.rnn(inputnew, (hn, cn))
        predG = torch.cat(predG,1)
        return predG, ExtG

if __name__=='__main__':
    vec = pd.read_csv('./Dietary_mergevec.csv')
    Dietary = pd.read_csv('./DietaryRes.csv')
    DietaryInfo = Dietary[['Heat','Carbohydrate','Fat','Protein','DietaryFiber']]
    beforemealG = Dietary[['beforemeal_%d'%t for t in range(4)]]
    aftermealG = Dietary[['postmeal_%d'%t for t in range(8)]]

    testtable = pd.read_csv('./Dietary_test.csv')
    testtable.columns=range(len(testtable.columns))
    MSEres = pd.DataFrame(index=range(CVtime),columns=['test_MSE','all_MSE'])
    Rres = pd.DataFrame(index=range(CVtime),columns=range(8))

    if Perturb:
        Dietarytest = DietaryInfo.copy()
        if Perturb == 'RawHeat':
            Dietarytest['Carbohydrate'] = Dietarytest['Heat'] * Ratio[0] / sum(Ratio) / 4
            Dietarytest['Protein'] = Dietarytest['Heat'] * Ratio[1] / sum(Ratio) / 4
            Dietarytest['Fat'] = Dietarytest['Heat'] * Ratio[2] / sum(Ratio) / 9
            if Fiber == 'H':
                Dietarytest['DietaryFiber'] = Dietarytest['Carbohydrate'] * 0.03
            elif Fiber == 'L':
                Dietarytest['DietaryFiber'] = Dietarytest['Carbohydrate'] * 0.06
        elif Perturb == 'HighHeat':
            Ratio = [5, 2, 3]
            Dietarytest['Carbohydrate'] = Dietarytest['Heat'] * Ratio[0] / sum(Ratio) / 4
            Dietarytest['Protein'] = Dietarytest['Heat'] * Ratio[1] / sum(Ratio) / 4
            Dietarytest['Fat'] = Dietarytest['Heat'] * Ratio[2] / sum(Ratio) / 9
            #Dietarytest['DietaryFiber'] = Dietarytest['Carbohydrate'] * 0.045
            Dietarytest = Dietarytest * 1.2
        elif Perturb == 'Standard':
            Ratio = [5, 2, 3]
            Dietarytest['Carbohydrate'] = Dietarytest['Heat'] * Ratio[0] / sum(Ratio) / 4
            Dietarytest['Protein'] = Dietarytest['Heat'] * Ratio[1] / sum(Ratio) / 4
            Dietarytest['Fat'] = Dietarytest['Heat'] * Ratio[2] / sum(Ratio) / 9
            #Dietarytest['DietaryFiber'] = Dietarytest['Carbohydrate'] * 0.045
        vec = vec.append(vec, ignore_index=True)
        beforemealG = beforemealG.append(beforemealG, ignore_index=True)
        aftermealG = aftermealG.append(aftermealG, ignore_index=True)

        testtable = testtable.append(testtable, ignore_index=True)
        testtable[0] = [False]*len(Dietarytest)+[True]*len(Dietarytest)
        CVtime = 1

        Scaler = MinMaxScaler()
        Scaler.fit(DietaryInfo.copy())
        DietaryInfo = DietaryInfo.append(Dietarytest, ignore_index=True)
        DietaryInfo = Scaler.transform(DietaryInfo)
    else:
        DietaryInfo = minmax_scale(DietaryInfo)
    minG = min(beforemealG.min().min(), aftermealG.min().min())
    maxG = max(beforemealG.max().max(), aftermealG.max().max())
    beforemealG = (beforemealG - minG) / (maxG - minG)
    aftermealG = (aftermealG - minG) / (maxG - minG)

    n = 100
    sched = PredictMetrics.cosine(n)
    lrs = [sched(t, 1) for t in range(n * 4)]
    print(CVtime)
    for testid in range(CVtime):
        X = np.concatenate([vec,DietaryInfo,beforemealG],axis=1)
        Y = aftermealG
        print(X.shape)
        print(Y.shape)
        train_ds, valid_ds, test_ds, all_ds = PredictMetrics.create_datasets(np.array(X),np.array(Y),testtable[testid])
        print(f'Creating data loaders with batch size: {bs}')
        trn_dl, val_dl = PredictMetrics.create_loaders(train_ds, valid_ds, bs)
        iterations_per_epoch = len(trn_dl)
        best_mse = 10000
        patience, trials = 100, 0
        mselist = []
        model = GlucosePredict(latentdim, layer_dim)
        model = model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = PredictMetrics.CyclicLR(opt, PredictMetrics.cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))
        criterion = nn.MSELoss()
        print('Start model training')
        if NeedTrain:
            for epoch in range(1, n_epochs + 1):
                for i, (x_batch, y_batch) in enumerate(trn_dl):
                    model.train()
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                    opt.zero_grad()
                    PreG, ExtG = model(x_batch)
                    loss = criterion(PreG, y_batch) + criterion(ExtG, x_batch[:, 128+5+1:])
                    loss.backward()
                    opt.step()
                    sched.step()
                model.eval()
                pred, y_true = [], []
                for x_val, y_val in val_dl:
                    x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                    PreG, ExtG = model(x_val)
                    pred=pred+PreG.tolist()
                    y_true=y_true+y_val.tolist()
                MSE = mean_squared_error(pred, y_true)
                mselist.append(MSE)
                overallMSE = MSE
                if overallMSE < best_mse:
                    trials = 0
                    best_mse = overallMSE
                    torch.save(model.state_dict(), '%s.pth'%modelpath)
                    print(f'\rEpoch {epoch} best model saved with MSE: {overallMSE:.4f}', end='')
                else:
                    trials += 1
                    if trials >= patience:
                        print(f'\nEarly stopping on epoch {epoch}')
                        break
        print('The training is finished! Restoring the best model weights')
        model.load_state_dict(torch.load('%s.pth'%modelpath))
        model.eval()
        test_dl = PredictMetrics.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)
        print('Predicting on test dataset')

        pred, y_true = [], []
        for x_val, y_val in test_dl:
            x_val, y_val = [t.cuda() for t in (x_val, y_val)]
            PreG, ExtG = model(x_val)
            pred = pred + PreG.tolist()
            y_true = y_true + y_val.tolist()
        pred = np.array(pred)
        y_true = np.array(y_true)
        MSE = mean_squared_error(pred, y_true)
        MSEres.loc[testid,'test_AllMSE']=MSE
        print('MSE_test=%.4f'%MSE)
        for j in range(8):
            Rres.loc[testid,j]=pearsonr(pred[:,j],y_true[:,j])[0]
            print(j,Rres.loc[testid,j])
        if recordresult:
            pd.DataFrame(np.concatenate([beforemealG[testtable[testid]], y_true, pred], axis=1) * (maxG - minG) + minG).to_csv(resultfile)

        pred, y_true = [], []
        all_dl = PredictMetrics.DataLoader(all_ds, batch_size=bs, shuffle=False, num_workers=4)
        for x_val, y_val in all_dl:
            x_val, y_val = [t.cuda() for t in (x_val, y_val)]
            PreG, ExtG = model(x_val)
            pred = pred + PreG.tolist()
            y_true = y_true + y_val.tolist()
        pred = np.array(pred)
        y_true = np.array(y_true)
        MSE = mean_squared_error(pred, y_true)
        MSEres.loc[testid, 'all_AllMSE'] = MSE
        print('MSE_all=%.4f'%MSE)
        pred = pred * (maxG - minG) + minG
        y_true = y_true * (maxG - minG) + minG

    print(MSEres['all_AllMSE'].values)
    print(MSEres['test_AllMSE'].values)
    Rres.to_csv('%s_Rres.csv'%modelpath)