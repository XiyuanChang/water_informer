import pandas as pd
import numpy as np
import os
from torch import from_numpy

root_dir = "../climate_new"

test_dates = []
with open(os.path.join(root_dir, "split_datesA.txt"), "r") as f:
    for line in f.readlines():
        date, mode = line.split(" ")
        date = date.strip()
        mode = mode.strip()
        if mode == "test":
            test_dates.append(pd.Timestamp(date.strip()))
    print(len(test_dates))


def count(idx):
    count = 0
    tcount = 0
    df_station = pd.read_csv(os.path.join(root_dir, "static_filtered.csv"), dtype={'STAID': str})
    df_station = df_station.drop(columns=['HUC02', 'HGAC'])
    df_station['STAID'] = df_station['STAID'].apply(lambda x: x.zfill(8))
    
    df_station_array = df_station.iloc[:, 1:].values
    max = df_station_array.max(axis=0)
    min = df_station_array.min(axis=0)
    df_station_array = (df_station_array - min) / (max - min)
    if np.isnan(df_station_array).sum() != 0:
        print("NaN in station data")
    
    station = df_station.loc[df_station['STAID'] == idx]
    station = from_numpy(station.iloc[:, 1:].values)
    # print(station.shape)
    # print(station)
    
    df = pd.read_csv(os.path.join(root_dir, f"{idx}.csv"))
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    
    station = df_station.loc[df_station['STAID'] == idx]
    station = station.iloc[:, 1:].values
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    col_names = df.columns[1:21]
    
    valid_count = np.zeros([20])
    
    for current_date in test_dates:
        df_date = df[df['Date'] == current_date]
        if df_date.shape[0] == 0:
            continue
        x_date = df_date.iloc[:, 21:].values
        if np.isnan(x_date).any():
            pass
        else:
            tcount += 1
        
        start_date = current_date - pd.Timedelta(days=365 - 1)
        mask = (df['Date'] >= start_date) & (df['Date'] <= current_date)
        data = df.loc[mask].copy()
        
        data_array = data.iloc[:, 1:].values

        # Handle case where the sequence is shorter than expected
        current_length = data_array.shape[0]
        if current_length < 365:
            pad_size = 365 - current_length
            # Create padding with NaN for missing data points
            pad = np.full((pad_size, data_array.shape[1]), np.nan)
            # Concatenate pad and data
            data_array = np.vstack((pad, data_array))

        x = from_numpy(data_array[:, 20:]).float()
        y = from_numpy(data_array[:, :20]).float()
        
        x = x[-1, :]
        y = y[-1, :].numpy()

        if x.isnan().any().item():
            pass
        else:
            valid_count += ~np.isnan(y)
            count += 1
    print(count)
    print(tcount)
    # get the index where valid_count < 51 and print the column name
    print(col_names[valid_count < 51])
    print(valid_count)
        

if __name__ == "__main__":
    idx = '14241500'
    # idx = '14161500'
    count(idx)