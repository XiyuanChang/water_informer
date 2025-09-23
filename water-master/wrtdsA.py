import pandas as pd
import numpy as np
from datetime import timedelta
import os
import statsmodels.api as sm
import time
import pickle
import json
import torch
from model.metric import kge, r_squared, pbias
from model.loss import mse_loss

def rmNan(Ls):
    # Initialize a set of all possible indices from the first array, assuming all arrays have the same shape
    if not Ls:
        return np.array([], dtype=int)

    valid_indices = set(range(Ls[0].shape[0]))

    # Iterate over each array in the list
    for arr in Ls:
        # Find indices in each row that contain NaN
        nan_indices = {idx for idx in range(arr.shape[0]) if np.isnan(arr[idx]).any()}
        
        # Remove indices where any NaN is found from the set of valid indices
        valid_indices -= nan_indices

        # If no valid indices remain, break early
        if not valid_indices:
            break
    idx = np.array(sorted(valid_indices), dtype=int)
    return [x[idx] for x in Ls], idx

h = [7, 2, 0.5]
the = 100
sn = 1e-5
log_every = 5
save_path = "saved_wrtdsA"
valid_threshold = 51
test_threshold = 2

train_dfs = {}
test_dfs = {}
train_dates, test_dates = [], []

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open("../climate_washed/split_datesA.txt", "r") as f:
    for line in f.readlines():
        date, mode = line.split(" ")
        date = date.strip()
        mode = mode.strip()
        
        if mode == 'train':
            train_dates.append(date)
        else:
            test_dates.append(date)

df_station = pd.read_csv("../climate_washed/static_filtered.csv", dtype={'STAID': str})
for staid in df_station['STAID']:
    if len(staid) < 8:
        staid_str = '0' * (8 - len(staid)) + staid
    else:
        staid_str = staid

    df = pd.read_csv(f"../climate_washed/{staid_str}.csv")
    train_df = df[df['Date'].isin(train_dates)]
    train_dfs[staid_str] = train_df

    test_df = df[df['Date'].isin(test_dates)]
    test_dfs[staid_str] = test_df


df_train = pd.concat(train_dfs)
# Calculate WRTDS from train and test set
varX = ['00060', 'sinT', 'cosT', 'datenum']
x_train = df_train[varX].values
df_train['Date'] = pd.to_datetime(df_train['Date'])
t_train = (df_train['Date'].dt.year + (df_train['Date'].dt.dayofyear / 365)).values

varY =  [
    '00010',
    '00095',
    '00300',
    '00400',
    '00405',
    '00600',
    '00605',
    '00618',
    '00660',
    '00665',
    '00681',
    '00915',
    '00925',
    '00930',
    '00935',
    '00940',
    '00945',
    '00955',
    '71846',
    '80154',
]
# y_train = df_train[varY].values

# q1 = x_train[:, 0].copy()
# q1[q1 < 0] = 0
# logq1 = np.log(q1+sn)
# x_train[:, 0] = logq1
# [xx1, yy1], ind1 = rmNan([x_train, y_train])

t0 = time.time()

targets = []
preds = []
kge_dict, r2_dict, pbias_dict = {}, {}, {}
count = 0

for id, test_df in test_dfs.items():
    if test_df.shape[0] == 0:
        kge_dict[id] = np.full(len(varY), np.nan)
        r2_dict[id] = np.full(len(varY), np.nan)
        pbias_dict[id] = np.full(len(varY), np.nan)
        continue
    
    count += 1
    # if os.path.exists(os.path.join(save_path, f"{id}.pkl")):
    #     print(f"Skipping {id} [{count}/{len(test_dfs)}]")
    #     state = pickle.load(open(os.path.join(save_path, f"{id}.pkl"), "rb"))

    #     # skip the rows with less than certain valid values
    #     # valid_count = np.sum(~np.isnan(test_df[varY].values), axis=0)
    #     # skip_index = np.where(valid_count < valid_threshold)[0]
    #     # import pdb; pdb.set_trace()
    #     target = torch.from_numpy(test_df[varY].values)
    #     # target[:, skip_index] = np.nan
    #     pred = state['pred']
    #     # pred[:, skip_index] = np.nan
        
    #     valid_count = (~torch.isnan(target)).sum(axis=0)
    #     full_kge = kge(pred, target)
    #     full_pbias = pbias(pred, target)
    #     full_r2 = r_squared(pred, target)
        
    #     full_kge = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_kge)
    #     full_pbias = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_pbias)
    #     full_r2 = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_r2)
        
    #     kge_dict[id] = full_kge.numpy()
    #     r2_dict[id] = full_r2.numpy()
    #     pbias_dict[id] = full_pbias.numpy()
    #     # print(pbias_dict[id])
    #     continue

    train_df = train_dfs[id]
    if train_df.shape[0] == 0:
        print("No training data for ", id)
        kge_dict[id] = np.full(len(varY), np.nan)
        r2_dict[id] = np.full(len(varY), np.nan)
        pbias_dict[id] = np.full(len(varY), np.nan)
        continue
    
    # prepare training data
    x_train = train_df[varX].values
    train_df['Date'] = train_df['Date'].apply(lambda x: pd.to_datetime(x))
    t_train = (train_df['Date'].dt.year + (train_df['Date'].dt.dayofyear / 365)).values
    y_train = train_df[varY].values

    q1 = x_train[:, 0].copy()
    q1[q1 < 0] = 0
    logq1 = np.log(q1+sn)
    x_train[:, 0] = logq1

    x_test = test_df[varX].values
    y_test = test_df[varY].values
    yOut = np.full([x_test.shape[0], len(varY)], np.nan)

    q2 = x_test[:, 0].copy()
    q2[q2 < 0] = 0
    logq2 = np.log(q2+sn)
    x_test[:, 0] = logq2
    test_df['Date'] = test_df['Date'].apply(lambda x: pd.to_datetime(x))
    t_test = (test_df['Date'].dt.year + (test_df['Date'].dt.dayofyear / 365)).values
    print('Testing station id:{} Time elapsed: {} Progress: [{}/{}]'.format(id, timedelta(seconds=int(time.time() - t0)), count, len(test_dfs)))
    
    # valid_count = np.sum(~np.isnan(y_test), axis=0)
    # skip_index = np.where(valid_count < valid_threshold)[0]
    # y_test[:, skip_index] = np.nan

    for indC, _ in enumerate(varY):
        # if indC in skip_index:
        #     yOut[:, indC] = np.nan
        #     continue
        y_train_temp = y_train[:, indC].copy()
        [xx1, _], ind1 = rmNan([x_train, y_train_temp])
        yy1 = y_train[ind1]

        if len(ind1) < test_threshold:
            # print("Not enough training data for ", id, " at ", varY[indC], " with ", len(ind1), " samples")
            y_test[:, indC] = np.nan
            # not enough training data, skip, see https://github.com/fkwai/geolearn/blob/1a5f54bb038e95fe091b666dbdc225f003a637a3/hydroDL/app/waterQuality/WRTDS.py#L170
            continue
        
        for k in range(x_test.shape[0]):
            if np.isnan(y_test[k, indC]):
                continue

            dY = np.abs(t_test[k] - t_train[ind1])
            dQ = np.abs(logq2[k] - logq1[ind1])
            dS = np.min(
                np.stack([abs(np.ceil(dY)-dY), abs(dY-np.floor(dY))]), axis=0)
            d = np.stack([dY, dQ, dS])
            n = d.shape[1]
            if n > the:
                hh = np.tile(h, [n, 1]).T
                bW = False
                while ~bW:
                    bW = np.sum(np.all(hh-d > 0, axis=0)) > the
                    hh = hh*1.1 if not bW else hh
            else:
                htemp = np.max(d, axis=1)*1.1
                hh = np.repeat(htemp[:, None], n, axis=1)
            w = (1-(d/hh)**3)**3
            w[w < 0] = 0
            wAll = w[0]*w[1]*w[2]
            ind = np.where(wAll > 0)[0]
            ww = wAll[ind]

            model = sm.WLS(yy1[ind][:, indC], xx1[ind], weights=ww).fit()
            yp = model.predict(x_test[k, :])[0]
            yOut[k, indC] = yp

    # targets.append(y_test)
    # preds.append(yOut)
    # with open(os.path.join(save_path, f"{id}.pkl"), "wb") as f:
    #     pickle.dump({'pred': yOut}, f)
    # loss_ = mse_loss(torch.from_numpy(yOut), torch.from_numpy(y_test), reduction='mean')
    # print("Loss: ", loss_.item())

    valid_count = (~torch.isnan(torch.from_numpy(y_test))).sum(axis=0)
    full_kge = kge(yOut, y_test)
    full_pbias = pbias(yOut, y_test)
    full_r2 = r_squared(yOut, y_test)
    
    full_kge = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_kge)
    full_pbias = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_pbias)
    full_r2 = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_r2)
    
    kge_dict[id] = full_kge.numpy()
    r2_dict[id] = full_r2.numpy()
    pbias_dict[id] = full_pbias.numpy()

col_names = ['staid']
for val in varY:
    col_names.append(val + "_kge")
    col_names.append(val + "_r2")
    col_names.append(val + "_pbias")

metric_df = pd.DataFrame(columns=col_names)
# metric_df['staid'] = list(kge_dict.keys())
metric_df['staid'] = df_station['STAID'].apply(lambda x: x.zfill(8))
for k, v in kge_dict.items():
    for i, name in enumerate(varY):
        metric_df.loc[metric_df['staid'] == k, f"{name}_kge"] = v[i]
        metric_df.loc[metric_df['staid'] == k, f"{name}_r2"] = r2_dict[k][i]
        metric_df.loc[metric_df['staid'] == k, f"{name}_pbias"] = pbias_dict[k][i]
metric_df.to_csv("wrtdsA_metric.csv", index=False, na_rep='')

# print("Final NSE: ", np.nanmean(kge_dict, axis=0))
# print("Final R2: ", np.nanmean(r2_dict, axis=0))
# print("Final PBIAS: ", np.nanmean(pbias_dict, axis=0))

# with open("wrtdsA_result.json", "w") as f:
#     d = {
#         "NSE": kge_dict,
#         "R2": r2_dict,
#         "PBIAS": pbias_dict
#     }
#     for k, v in d.items():
#         d[k] = [x.tolist() for x in v]
#     json.dump(d, f, indent=4)
    
    
# with open("wrtdsA_result.txt", "w") as f:
#     f.write(f"NSE: {np.nanmean(kge_dict, axis=0)}\n")
#     f.write(f"R2: {np.nanmean(r2_dict, axis=0)}\n")
#     f.write(f"PBIAS: {np.nanmean(pbias_dict, axis=0)}\n")