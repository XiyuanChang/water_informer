import pandas as pd
import os
import json

path = "lstm_ablation/group_1/models/lstm/0712_152220"

split_dict = {
    "ClimateDatasetV2A": "split_datesA.txt",
    "ClimateDatasetV2B": "split_datesB.txt",
    "ClimateDatasetV2C": "split_datesC.txt"
}

metric_path = os.path.join(path, "predictions/metric/metrics.csv")
config_path = os.path.join(path, "config.json")

config = json.load(open(config_path))
split_path = os.path.join("../climate_new", split_dict[config["dataset"]])
y_feature = config["y_feature"]
columns = ['staid']

metric_df = pd.read_csv(metric_path, dtype={"staid": str})

# load test dates
test_dates = []
with open(split_path, "r") as f:
    for line in f.readlines():
        date, mode = line.strip().split(" ")
        date = date.strip()
        mode = mode.strip()
        
        if mode == 'test':
            test_dates.append(date)

for staid in metric_df["staid"]:
    sta_id = staid.zfill(8)
    
    df = pd.read_csv(f"../climate_new/{sta_id}.csv")
    
    mask = df.iloc[:, 21:].notna().all(axis=1)
    mask &= df["Unnamed: 0"].isin(test_dates)
    
    for name in y_feature:
        test_mask = mask & df.loc[:, name].notna()
        num = test_mask.sum()
        metric_df.loc[metric_df["staid"] == staid, f"{name}_num"] = num

for name in y_feature:
    columns.append(f"{name}_kge")
    columns.append(f"{name}_r2")
    columns.append(f"{name}_pbias")
    columns.append(f"{name}_num")

metric_df = metric_df[columns]

save_path = os.path.join(path, "predictions/metric/metric_num.csv")
print("Metric new csv saved to ", save_path)
metric_df.to_csv(save_path, index=False, na_rep='')