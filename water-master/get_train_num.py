import pandas as pd
import os
import json


split_dict = {
    "ClimateDatasetV2A": "split_datesA.txt",
    "ClimateDatasetV2B": "split_datesB.txt",
    "ClimateDatasetV2C": "split_datesC.txt"
}

split_path = os.path.join("../climate_new", split_dict["ClimateDatasetV2C"])
station_df = pd.read_csv("../climate_new/static_filtered.csv")
y_feature = ["00010", "00095", "00300", "00400", "00405", "00600", "00605", "00618", "00660", "00665", "00681", "00915", "00925", "00930", "00935", "00940", "00945", "00955", "71846", "80154"]

columns = ['staid']
for name in y_feature:
    columns.append(f"{name}_num")

new_df = pd.DataFrame(columns=columns)
new_df['staid'] = station_df['STAID']

# load train dates
train_dates = []
with open(split_path, "r") as f:
    for line in f.readlines():
        date, mode = line.strip().split(" ")
        date = date.strip()
        mode = mode.strip()
        
        if mode == 'test':
            train_dates.append(date)

for staid in station_df["STAID"]:
    sta_id = str(staid).zfill(8)
    
    df = pd.read_csv(f"../climate_new/{sta_id}.csv")
    # import pdb; pdb.set_trace()
    # mask = df["Date"].isin(train_dates)
    mask = df.iloc[:, 0].isin(train_dates)

    for name in y_feature:
        train_mask = mask & df.loc[:, name].notna()
        num = train_mask.sum()
        new_df.loc[new_df["staid"] == staid, f"{name}_num"] = num

new_df.to_csv("test_num.csv", index=False, na_rep='')