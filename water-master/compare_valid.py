import pandas as pd
import numpy as np
import os
from torch import from_numpy

root_dir = "../climate_washed"

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
    
    df = pd.read_csv(os.path.join(root_dir, f"{idx}.csv"))
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    col_names = df.columns[1:21]
    
    df = df.loc[~(df.iloc[:, 21:].isna().any(axis=1)), :]
    
    target = df.iloc[:, 1:21].values
    print((~np.isnan(target)).sum(axis=0))
    
    df = df.loc[~(df.iloc[:, 1:21].isna().all(axis=1)), :]
    target = df.iloc[:, 1:21].values
    print((~np.isnan(target)).sum(axis=0))



if __name__ == "__main__":
    idx = '14241500'
    # idx = '14161500'
    count(idx)