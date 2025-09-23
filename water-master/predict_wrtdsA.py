import pandas as pd
import numpy as np
from datetime import timedelta
import os
import statsmodels.api as sm
import time
import copy
import pickle
from tqdm import tqdm
import json
from model.metric import kge, r_squared, pbias

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
save_path = "save_wrtdsA"

train_dfs = []
test_dfs = {}
train_dates, test_dates = [], []

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

    df_washed = pd.read_csv(f"../climate_washed/{staid_str}.csv")
    df_washed.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    train_df = df_washed[df_washed['Date'].isin(train_dates)]
    
    if train_df.shape[0]:
        train_dfs.append(train_df)
    
    df = pd.read_csv(f"../climate_new/{staid_str}.csv")
    test_dfs[staid_str] = df.copy()

    # test_df = df[df['Date'].isin(test_dates)]
    # if test_df.shape[0]:
    #     test_dfs[staid_str] = test_df


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
y_train = df_train[varY].values

q1 = x_train[:, 0].copy()
q1[q1 < 0] = 0
logq1 = np.log(q1+sn)
x_train[:, 0] = logq1
[xx1, yy1], ind1 = rmNan([x_train, y_train])

t0 = time.time()

targets = []
preds = []
nse_list, r2_list, pbias_list = [], [], []
count = 0

for id, test_df in test_dfs.items():
    count += 1
    if os.path.exists(os.path.join(save_path, f"{id}.csv")):
        print(f"Skipping {id} [{count}/{len(test_dfs)}]")
        
        # state = pickle.load(open(os.path.join(save_path, f"{id}.pkl"), "rb"))
        # preds.append(state['pred'])
        # targets.append(test_df[varY].values)
        # nse_list.append(nse(state['pred'], test_df[varY].values).numpy())
        # r2_list.append(r_squared(state['pred'], test_df[varY].values).numpy())
        # pbias_list.append(pbias(state['pred'], test_df[varY].values).numpy())
        continue
    
    test_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    x_test = test_df[varX].values
    y_test = test_df[varY].values
    yOut = np.full([x_test.shape[0], len(varY)], np.nan)
    
    q2 = x_test[:, 0].copy()
    q2[q2 < 0] = 0
    logq2 = np.log(q2+sn)
    x_test[:, 0] = logq2
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    t_test = (test_df['Date'].dt.year + (test_df['Date'].dt.dayofyear / 365)).values
    
    output_df = copy.deepcopy(test_df)
    # add columns to prediction csv
    for name in varY:
        output_df[f"{name}_pred"] = np.nan
    for name in varY:
        output_df[f"{name}_type"] = 3

    print('Testing station id:{} Time elapsed: {} Progress: [{}/{}]'.format(id, timedelta(seconds=int(time.time() - t0)), count, len(test_dfs)))
    
    for indC, name in enumerate(varY):        
        
        for k in tqdm(range(x_test.shape[0])):
            if np.isnan(x_test[k, :]).any():
                # x 不全 type=3                
                continue
                        
            if np.isnan(y_test[k, indC]):
                # y 不全 x齐全 type=2
                output_df.loc[k, f"{name}_type"] = 2
            else:
                date = output_df.iloc[k, 0].strftime("%Y-%m-%d")
                
                if str(date) in train_dates:
                    output_df.loc[k, f"{name}_type"] = 0
                elif str(date) in test_dates:
                    output_df.loc[k, f"{name}_type"] = 1
                else:
                    print("Warning: date {} not found in station {}".format(date, id))
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
            output_df.loc[k, f"{name}_pred"] = yp
            # yOut[k, indC] = yp

    with open(os.path.join(save_path, f"{id}.csv"), "w") as f:
        output_df.to_csv(f, index=False, na_rep='')
    # targets.append(y_test)
    # preds.append(yOut)
    # with open(os.path.join(save_path, f"{id}.pkl"), "wb") as f:
    #     pickle.dump({'pred': yOut}, f)

#     nse_list.append(nse(yOut, y_test).numpy())
#     r2_list.append(r_squared(yOut, y_test).numpy())
#     pbias_list.append(pbias(yOut, y_test).numpy())

# print("Final NSE: ", np.nanmean(nse_list, axis=0))
# print("Final R2: ", np.nanmean(r2_list, axis=0))
# print("Final PBIAS: ", np.nanmean(pbias_list, axis=0))

# with open("wrtdsA_result.json", "w") as f:
#     d = {
#         "NSE": nse_list,
#         "R2": r2_list,
#         "PBIAS": pbias_list
#     }
#     for k, v in d.items():
#         d[k] = [x.tolist() for x in v]
#     json.dump(d, f, indent=4)
    
    
# with open("wrtdsA_result.txt", "w") as f:
#     f.write(f"NSE: {np.nanmean(nse_list, axis=0)}\n")
#     f.write(f"R2: {np.nanmean(r2_list, axis=0)}\n")
#     f.write(f"PBIAS: {np.nanmean(pbias_list, axis=0)}\n")