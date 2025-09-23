import pandas as pd
import numpy as np
# for deeponet
columns = ["STAID"]
with open("climate_washed/01054200.csv", "r") as f:
    station_df = pd.read_csv(f)
    columns.extend(station_df.columns[1:21])
    print(columns)

train_result_df = pd.DataFrame(columns=columns)
test_result_df = pd.DataFrame(columns=columns)
train_dates, test_dates = [], []
with open("climate_washed/split_datesC.txt", "r") as f:
    for l in f.readlines():
        date, split = l.split(" ")
        date = date.strip()
        split = split.strip()
        if split == "train":
            train_dates.append(date)
        else:
            test_dates.append(date)

with open("climate_washed/static_filtered.csv", "r") as f:
    station_df = pd.read_csv(f)
    # zfill the STAID field
    station_df['STAID'] = station_df['STAID'].apply(lambda x: str(x).zfill(8))
    train_result_df['STAID'] = station_df['STAID']
    test_result_df['STAID'] = station_df['STAID']

for staid in station_df['STAID']:
    station_df = pd.read_csv(f"climate_washed/{staid}.csv")
    train_df = station_df[station_df['Date'].isin(train_dates)]
    test_df = station_df[station_df['Date'].isin(test_dates)]
    
    train_y = train_df.iloc[:, 1:21].values
    # import pdb; pdb.set_trace()
    # check if train_y contains non-null values in every column
    y = (~np.isnan(train_y)).any(axis=0).astype(int)
    train_result_df.loc[train_result_df['STAID'] == staid, columns[1:21]] = y

    test_y = test_df.iloc[:, 1:21].values
    y = (~np.isnan(test_y)).any(axis=0).astype(int)
    test_result_df.loc[test_result_df['STAID'] == staid, columns[1:21]] = y

train_result_df.to_csv("statistics/train_deeponet.csv", index=False)
test_result_df.to_csv("statistics/test_deeponet.csv", index=False)

# for lstm
train_result_df = pd.DataFrame(columns=columns)
test_result_df = pd.DataFrame(columns=columns)
train_dates, test_dates = [], []
with open("climate_new/split_datesC.txt", "r") as f:
    for l in f.readlines():
        date, split = l.split(" ")
        date = date.strip()
        split = split.strip()
        if split == "train":
            train_dates.append(date)
        else:
            test_dates.append(date)

with open("climate_new/static_filtered.csv", "r") as f:
    station_df = pd.read_csv(f)
    # zfill the STAID field
    station_df['STAID'] = station_df['STAID'].apply(lambda x: str(x).zfill(8))
    train_result_df['STAID'] = station_df['STAID']
    test_result_df['STAID'] = station_df['STAID']

for staid in station_df['STAID']:
    station_df = pd.read_csv(f"climate_new/{staid}.csv")
    train_df = station_df[station_df['Unnamed: 0'].isin(train_dates)]
    test_df = station_df[station_df['Unnamed: 0'].isin(test_dates)]
    
    train_y = train_df.iloc[:, 1:21].values
    # import pdb; pdb.set_trace()
    # check if train_y contains non-null values in every column
    y = (~np.isnan(train_y)).any(axis=0).astype(int)
    train_result_df.loc[train_result_df['STAID'] == staid, columns[1:21]] = y

    test_y = test_df.iloc[:, 1:21].values
    y = (~np.isnan(test_y)).any(axis=0).astype(int)
    test_result_df.loc[test_result_df['STAID'] == staid, columns[1:21]] = y

train_result_df.to_csv("statistics/train_lstm.csv", index=False)
test_result_df.to_csv("statistics/test_lstm.csv", index=False)