import pandas as pd
import random

with open("climate_new_copy/static_filtered.csv", "r") as f:
    data = pd.read_csv(f)

stations = data['STAID'].tolist()
stations = list(map(lambda x: str(x).zfill(8), stations))

# randomly select 80% of the stations as train split, the rest into test
train_stations = random.sample(stations, int(len(stations) * 0.5))
test_stations = list(set(stations) - set(train_stations))

# sort the stations
train_stations.sort(key=lambda x: int(x))
test_stations.sort(key=lambda x: int(x))

with open("climate_new_copy/station_split.txt", "w") as f:
    for station in train_stations:
        f.write(station + " " + "train\n")
    for station in test_stations:
        f.write(station + " " + "test\n")

# with open("cluster_results.csv", "r") as f:
#     data = pd.read_csv(f, dtype={'STAID': str})

# df = data[data['Cluster'] == 3]

# # randomly sample 30 lines from data
# df = df.sample(n=30)

# test_ids = df['STATION_ID'].tolist()
# full_ids = data['STATION_ID']
# train_ids = list(set(full_ids) - set(test_ids))

# train_ids.sort(key=lambda x: int(x))
# test_ids.sort(key=lambda x: int(x))

# with open("climate_new/station_split.txt", "w") as f:
#     for station in train_ids:
#         f.write(str(station).zfill(8) + " " + "train\n")
#     for station in test_ids:
#         f.write(str(station).zfill(8) + " " + "test\n")
# with open("climate_washed/station_split.txt", "w") as f:
#     for station in train_ids:
#         f.write(str(station).zfill(8) + " " + "train\n")
#     for station in test_ids:
#         f.write(str(station).zfill(8) + " " + "test\n")
