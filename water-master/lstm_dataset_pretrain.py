import os
import pandas as pd
from torch.utils.data import Dataset
from torch import from_numpy
from mytransform import MeanStdNormalize, MinMaxNormalize, InverseLogMinMax, MinMaxDeNormalize, LogNormalize, LogDeNormalize, LogMinMax, CustomQuantileTransformer
import numpy as np
import pickle
import torch
from tqdm import tqdm

class ClimateDatasetV2(Dataset):
    def __init__(self, root_dir="../climate_new/", branch_transform=None, trunk_transform=None, target_transform=None, split='train'):
        """
        Args:
            root_dir (str): 
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        if split not in ['train', 'test', 'all']:
            raise ValueError('split should be one of train, val, test, all')
        df_station = pd.read_csv(os.path.join(root_dir, "static_filtered.csv"), dtype={'STAID': str})
        
        temporal = []
        targets = []
        stations = []

        split_path = os.path.join(root_dir, 'train_test_split.txt')
        with open(split_path, "r") as f:
            for line in f.readlines():
                idx, mode = line.split(" ")
                mode = mode.strip()
                
                if mode == split or split == 'all':
                    df = pd.read_csv(os.path.join(root_dir, f"{idx}.csv"))
                    temporal_data = df.iloc[:, 21:].values
                    temporal.append(temporal_data)

                    target = df.iloc[:, 1:21].values
                    
                    targets.append(target)
                    
                    staid = idx.lstrip('0')
                    station = df_station.loc[df_station['STAID'] == staid]
                    station = station.iloc[:, 1:].values
                    station = np.repeat(station, temporal_data.shape[0], axis=0)
                    assert station.shape[0] == target.shape[0]
                    stations.append(station)
        
        self.stations = from_numpy(np.concatenate(stations, axis=0)).float()

        targets = np.concatenate(targets, axis=0)
        mask = ~np.isnan(targets)
        tensor = from_numpy(targets[mask]).float()
        tensor_nan = torch.full(targets.shape, torch.nan, dtype=torch.float32)
        mask_flat = mask.flatten()
        tensor_nan_flat = tensor_nan.flatten()
        tensor_nan_flat[mask_flat] = tensor
        self.targets = tensor_nan_flat.reshape(targets.shape)

        self.temporal = from_numpy(np.concatenate(temporal, axis=0)).float()
        self.b_transforms = branch_transform
        self.t_transforms = trunk_transform
        self.target_transform = target_transform
        print(self.targets.shape)
        print(self.temporal.shape)
        print(self.stations.shape)
    
    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return self.targets.shape[0]
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        Args:
            idx (int): Index of the item in the dataset
        
        Returns:
            A sample from the dataset after applying the transformation.
        """
        temporal = self.temporal[idx]
        station = self.stations[idx]
        target = self.targets[idx]
        if self.b_transforms:
            temporal = self.b_transforms(temporal)
        if self.t_transforms:
            station = self.t_transforms(station)
        if self.target_transform:
            target = self.target_transform(target)
        return temporal, station, target


class ClimateDatasetV2A(ClimateDatasetV2):
    DATE_PATH = "split_datesA.txt"
    test_year = list(range(1985,2019))
    # test_year = [1985]

    def __init__(self, root_dir="../climate_new/", transform='MeanStdNormalize', split='train', stats=None, 
                 seqlen=365, testNum=None, x_feature=None, y_feature=None, exclude=0, retDate=False, minmax_feature=None):
        """
        Args:
            root_dir (str): 
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        print("Using data from directory: ", root_dir)
        if split not in ['train', 'test', 'all', 'testsubset']:
            raise ValueError('split should be one of train, val, test, all')
        self.split = split
        self.seqlen = seqlen
        
        self.retDate = retDate
        self.x_feature = x_feature
        self.y_feature = y_feature
        
        df_station = pd.read_csv(os.path.join(root_dir, "static_filtered.csv"), dtype={'STAID': str})
        df_station['STAID'] = df_station['STAID'].apply(lambda x: x.zfill(8))

        ### 加
        self.station_colnames = df_station.columns[1:]   # 记住静态特征列名（不含 STAID）

        # df_station = df_station.drop(columns=['SNOW_PCT_PRECIP'])
        # df_station.drop(columns=['HGAC'], inplace=True)
        # if split is 'test', randomly select 50 stations for testing
        if split == 'test' and testNum:
            df_station = df_station.sample(n=testNum, random_state=42)
        elif split == 'all' and testNum:
            df_station = df_station.sample(n=testNum, random_state=42)
        # else:
        #     df_station = df_station.sample(n=100, random_state=42)

        if exclude:
            import json
            with open("../group.json", 'r') as f:
                group = json.load(f)
            if isinstance(exclude, int):
                exclude_features = group[str(exclude)]
            else:
                exclude_features = []
                for e in exclude:
                    exclude_features += group[str(e)]
        else:
            exclude_features = []
        if x_feature:
            x_feature = [col for col in x_feature if col not in exclude_features]
            self.x_feature = x_feature
        print("X features: ", self.x_feature)
        temporal = []
        targets = []
        self.sequences = []
        self.labels = []
        self.sample_info = []
        
        self.dates = []
        test_dates = []

        with open(os.path.join(root_dir, self.DATE_PATH), "r") as f:
            for line in f.readlines():
                date, mode = line.split(" ")
                date = date.strip()
                mode = mode.strip()
                if split == 'all':
                    self.dates.append(date)
                    self.sample_info.extend([(staid, pd.Timestamp(date.strip())) for staid in df_station['STAID']])
                
                if split == 'testsubset':
                    # check the year of the date
                    year = int(date.split("-")[0])
                    if year in self.test_year and mode == 'test':
                        self.dates.append(date)
                        self.sample_info.extend([(staid, pd.Timestamp(date.strip())) for staid in df_station['STAID']])

                if mode == split:
                    # split = 'train' or 'test'
                    self.dates.append(date)
                    self.sample_info.extend([(staid, pd.Timestamp(date.strip())) for staid in df_station['STAID']])

                if mode == 'test':
                    test_dates.append(date)

        print(len(self.sample_info))
        self.station_temporal = {}
        self.station_feature = {}
        print(f"Loading {split} Dataset...")
        for staid in tqdm(df_station['STAID']):
            staid_str = staid.zfill(8)
            df = pd.read_csv(os.path.join(root_dir, f"{staid_str}.csv"))
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            
            if split == "train":
                station = df_station.loc[df_station['STAID'] == staid]
                if exclude:
                    station = station.drop(columns=[col for col in exclude_features if col in station.columns])
                station = station.iloc[:, 1:].values
                self.station_feature[staid_str] = station

                # This is used for calculating statistics of train data (mean, std, etc.)
                ### 加
                # station 是 numpy，形状 [1, n_static]；转成 dict 方便取值
                station_cols = self.station_colnames.tolist()
                #print("Station columns: ", station_cols)
                #print('station features: ', self.station_feature.keys())
                station_vals = self.station_feature[staid_str][0]  # 该站静态特征的一行
                static_map = {col: station_vals[i] for i, col in enumerate(station_cols)}
                #print("Static map example: ", list(static_map.items())[:5])  # 打印前5个键值对看看
                ####################
                df_train = df[df['Date'].isin(self.dates)]
                if x_feature:
                    temporal_data = df_train.loc[:, x_feature].values
                else:
                    temporal_data = df_train.iloc[:, 21:].values
                temporal.append(temporal_data)

                if y_feature:
                    # 加：找出 y_feature 里不在 df_train 的列（这些应该都是静态列）
                    missing = [c for c in y_feature if c not in df_train.columns]
                    if len(missing) > 0:
                        # 用该站的静态值，把缺失列补到 df_train，每列值全相同
                        for c in missing:
                            if c not in static_map:
                                raise KeyError(f"Static feature '{c}' not found in static_filtered.csv")
                            df_train[c] = static_map[c]
                    target = df_train.loc[:, y_feature].values
                else:
                    target = df_train.iloc[:, 1:21].values
                targets.append(target)

                
                df.loc[df['Date'].isin(test_dates), df.columns[1:]] = None
                df['Date'] = pd.to_datetime(df['Date'])
                self.station_temporal[staid_str] = df

            else:
                station = df_station.loc[df_station['STAID'] == staid]
                if exclude:
                    station = station.drop(columns=[col for col in exclude_features if col in station.columns])
                   
                station = station.iloc[:, 1:].values
                self.station_feature[staid_str] = station
                
                df['Date'] = pd.to_datetime(df['Date'])
                self.station_temporal[staid_str] = df

       
        if minmax_feature:
            print(1)
            station_columns = df_station.columns
            if exclude:
                station_columns = station_columns.drop([col for col in exclude_features if col in station_columns])
            station_minmax_idx = station_columns.get_indexer(minmax_feature)
            # filter out -1 in idx
            station_minmax_idx = station_minmax_idx[station_minmax_idx != -1]
            # print(station.columns[station_minmax_idx]) # this yields ['LAT_GAGE', 'LNG_GAGE']
            station_minmax_idx -= 1
            self.station_minmax_idx = station_minmax_idx

            if x_feature:
                temporal_minmax_idx = [x_feature.index(col) for col in minmax_feature if col in x_feature]
            else:
                col = df.columns[21:]
                temporal_minmax_idx = col.get_indexer(minmax_feature)
                temporal_minmax_idx = temporal_minmax_idx[temporal_minmax_idx != -1]
            self.temporal_minmax_idx = temporal_minmax_idx

            if y_feature:
                self.y_minmax_idx = [y_feature.index(col) for col in minmax_feature if col in y_feature]
            else:
                col = df.columns[1:21]
                y_minmax_idx = col.get_indexer(minmax_feature)
                self.y_minmax_idx = y_minmax_idx[y_minmax_idx != -1]

    
        if split == 'train':
            self.temporal = from_numpy(np.concatenate(temporal, axis=0)).float()
            targets = np.concatenate(targets, axis=0)
            self.targets = from_numpy(targets).float()
            
            temp_stat = calculate_statistics(self.temporal)
            target_stat = calculate_statistics(self.targets)
            station_stat = self.get_station_stats()

        else:
            temp_stat = stats['temporal']
            target_stat = stats['target']
            station_stat = stats['station']

            # final_x, final_y = [], []
            
            # for staid in tqdm(df_station['STAID']):
            #     staid_str = staid.zfill(8)
            #     df = pd.read_csv(os.path.join(root_dir, f"{staid_str}.csv"))
            #     df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
                
            #     station = df_station.loc[df_station['STAID'] == staid]
            #     station = station.iloc[:, 1:].values

            #     mask = df['Date'].isin(self.dates)
            #     mask = mask.to_list()
            #     full_data = df.iloc[:, 1:].values

            #     for idx in mask:
            #         data_array = full_data[max(0, idx - self.seqlen + 1):idx + 1]
            #         current_length = data_array.shape[0]
            #         if current_length < self.seqlen:
            #             pad_size = self.seqlen - current_length
            #             pad = np.full((pad_size, data_array.shape[1]), np.nan)
            #             # Concatenate pad and data
            #             data_array = np.vstack((pad, data_array))

            #         x = from_numpy(data_array[:, 20:]).float()
            #         y = from_numpy(data_array[:, :20]).float()
                    
            #         x = self.x_transform(x)
            #         y = self.y_transform(y)
                    
            #         station_concat = from_numpy(np.repeat(station, x.shape[0], axis=0)).float()
            #         station_concat = self.station_transform(station_concat)
            #         x = torch.cat((x, station_concat), dim=1)

            #         x[torch.isnan(x)] = -1
            #         y[torch.isnan(y)] = -1

            #         final_x.append(x)
            #         final_y.append(y)
                    
            #     self.station_temporal[staid_str] = {"x": torch.stack(final_x), "y": torch.stack(final_x)}
        if transform == "MeanStdNormalize":
            self.x_transform = MeanStdNormalize(
                temp_stat['mean'],
                temp_stat['variance']
            )
            self.station_transform = MeanStdNormalize(
                station_stat['mean'],
                station_stat['variance']
            )
        elif transform == "MinMaxNormalize":
            self.x_transform = MinMaxNormalize(
                temp_stat['min'],
                temp_stat['max']
            )
            self.station_transform = MinMaxNormalize(
                station_stat['min'],
                station_stat['max']
            )
        elif transform == "MinMaxNormalizeReal":
            self.x_transform = MinMaxNormalize(
                temp_stat['max'],
                temp_stat['min']
            )
            self.station_transform = MinMaxNormalize(
                station_stat['max'],
                station_stat['min']
            )
        elif transform == "LogMinMax":
            self.x_transform = LogMinMax(
               temp_stat,
                index=temporal_minmax_idx
            )
            self.station_transform = LogMinMax(
                station_stat,
                index=station_minmax_idx
            )
        self.y_transform = LogNormalize(
            target_stat,
            index=self.y_minmax_idx
        )
        # print("Applying minmax to y features: ", self.y_minmax_idx)
        return

    def __len__(self):
        return len(self.sample_info)

    def get_station_stats(self):
        station = list(self.station_feature.values())
        station = from_numpy(np.concatenate(station, axis=0)).float()
        return calculate_statistics(station)

    def get_stats(self):
        return {
            'temporal': calculate_statistics(self.temporal),
            'target': calculate_statistics(self.targets),
            'station': self.get_station_stats()
        }

    def get_target_detransform(self):
        target_stat = calculate_statistics(self.targets)
        return LogDeNormalize(
            target_stat,
            self.y_minmax_idx
        )
    def get_target_transform(self):
        return self.y_transform
    
    def get_x_transform(self):
        return self.x_transform
    
    def get_station_transform(self):
        return self.station_transform
    

    def get_x_detransform(self):
        temp_stat = calculate_statistics(self.temporal)
        return InverseLogMinMax(
            temp_stat,
            self.temporal_minmax_idx
        )

    def __getitem__(self, idx):
        # if self.split == 'train':
        staid, current_date = self.sample_info[idx]
        
        staid_str = staid.zfill(8)  # Ensure station ID is zero-padded to correct length

        # Retrieve the preloaded data
        df = self.station_temporal[staid_str]
        # Find data from the required range
        start_date = current_date - pd.Timedelta(days=self.seqlen - 1)
        mask = (df['Date'] >= start_date) & (df['Date'] <= current_date)
        data = df.loc[mask].copy()

        # —— 补 y_feature 里需要的静态列（先保证 data 有完整列）——
        if hasattr(self, "station_colnames"):
            station_cols = list(self.station_colnames)
        else:
            # 兜底：从 station_feature 推断列顺序（你在 __init__ 里最好保存过 station_colnames）
            raise AttributeError("self.station_colnames 未定义，请在 __init__ 中保存 df_station.columns[1:] 到 self.station_colnames")

        station_vals = self.station_feature[staid_str][0]  # 1 x S
        static_map = {col: station_vals[i] for i, col in enumerate(station_cols)}

        missing = [c for c in self.y_feature if c not in data.columns]
        for c in missing:
            if c not in static_map:
                raise KeyError(f"Static feature '{c}' not found for station {staid_str}")
            data[c] = static_map[c]

        # 2) 组 x（仅动态 74 列；如果你 x_feature 里只放动态列，这里就简单）
        if self.x_feature:
            x = data.loc[:, self.x_feature].to_numpy()
        else:
            x = data.iloc[:, 21:].to_numpy()  # 原来的 74 动态特征

        cur_len = x.shape[0]
        if cur_len < self.seqlen:
            pad = self.seqlen - cur_len
            x = np.vstack((np.full((pad, x.shape[1]), np.nan), x))
        x = torch.from_numpy(x).float()

        # 3) 取 y_t（t 当天，94列）——用 reindex 保证列顺序
        row_t = data.tail(1).reindex(columns=self.y_feature)
        if len(row_t) == 0:
            y_t_np = np.full((len(self.y_feature),), np.nan, dtype=float)
        else:
            y_t_np = row_t.iloc[0].to_numpy(dtype=float)

        # 4) 取 y_next（t+1，94列）——从 df 取一行，然后同样补静态列，再按 y_feature 排列
        next_date = current_date + pd.Timedelta(days=1)
        row_next_df = df.loc[df['Date'] == next_date].copy()
        if len(row_next_df) > 0:
            # 给 next 行补静态列
            missing_next = [c for c in self.y_feature if c not in row_next_df.columns]
            for c in missing_next:
                if c not in static_map:
                    raise KeyError(f"Static feature '{c}' not found for station {staid_str}")
                row_next_df[c] = static_map[c]
            row_next_df = row_next_df.reindex(columns=self.y_feature)
            y_next_np = row_next_df.iloc[0].to_numpy(dtype=float)
        else:
            y_next_np = np.full((len(self.y_feature),), np.nan, dtype=float)

        # 5) 拼两帧 y，并只做一次归一化（保持 NaN）
        y_two = torch.from_numpy(np.stack([y_t_np, y_next_np], axis=0)).float()  # [2, 94]
        y_two = self.y_transform(y_two)  # 只做一次；不要对 y_t/y_next 单独做了

        # 6) 拼 station 到 x（每个时间步重复一次）
        station = self.station_feature[staid_str]
        station = from_numpy(np.repeat(station, x.shape[0], axis=0)).float()
        station = self.station_transform(station)

        x = self.x_transform(x)
        x = torch.cat((x, station), dim=1)

        # x 的缺失用 -1，y 仍保留 NaN（loss 会跳过 NaN）
        x[torch.isnan(x)] = -1

        if self.split == "train":
            return x, y_two
        elif self.split == "all":
            return staid_str, current_date.strftime('%Y-%m-%d'), x, y_two
        else:
            if self.retDate:
                return staid_str, current_date.strftime('%Y-%m-%d'), x, y_two
            return staid_str, x, y_two


def calculate_statistics(tensor):
    mask = torch.isnan(tensor)
    masked_tensor = torch.where(mask, torch.tensor(0.0, device=tensor.device), tensor)
    count = (~mask).sum(dim=0)
    
    # mean
    mean = masked_tensor.sum(dim=0) / count
    # var
    diff_squared = (tensor - mean)**2
    masked_diff_squared = torch.where(mask, torch.tensor(0.0, device=tensor.device), diff_squared)
    sum_diff_squared = masked_diff_squared.sum(dim=0)
    variance = sum_diff_squared / count
    # variance = torch.sqrt(variance)
    # min, max
    max = masked_tensor.max(dim=0)[0]
    min = masked_tensor.min(dim=0)[0]
    
    y = tensor.float()
    y_log = torch.log(y.clamp(min=1e-9))
    y_log_add = torch.log(y+1e-4)
    log_max = nanmax(y_log_add, dim=0)[0]
    log_min = nanmin(y_log_add, dim=0)[0]
    log_mean = torch.nanmean(y_log_add, dim=0)
    log_var = torch.nanmean(y_log_add, dim=0) # nan log var

    # log_tensor = torch.log(masked_tensor.clamp(min=1e-9))  # Apply clamping to handle zeros (though tensor is positive)
    # log_max = log_tensor.max(dim=0)[0]
    # log_min = log_tensor.min(dim=0)[0]
    
    perc10 = torch.nanquantile(y_log, 0.10, dim=0, keepdim=True)
    perc90 = torch.nanquantile(y_log, 0.90, dim=0, keepdim=True)

    scaler = CustomQuantileTransformer(output_distribution='uniform')
    scaler.fit(tensor.numpy())
    quantile_init_params = scaler.get_init_params()
    quantile_fit_params = scaler.get_fitted_params()

    scaler = CustomQuantileTransformer(output_distribution='uniform')
    scaler.fit(y_log.numpy())
    quantile_init_params_log = scaler.get_init_params()
    quantile_fit_params_log = scaler.get_fitted_params()
    
    return {'mean': mean, 'variance': variance, 'max': max, 'min': min, '10th': perc10, '90th': perc90,
            'logmax': log_max, 'logmin': log_min, 'quantile_init_params': quantile_init_params, 'quantile_fit_params': quantile_fit_params,
            'quantile_init_params_log': quantile_init_params_log, 'quantile_fit_params_log': quantile_fit_params_log,
            'log_mean': log_mean, 'log_var': log_var}

def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output

def nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output

class ClimateDatasetV2B(ClimateDatasetV2A):
    DATE_PATH = "split_datesB.txt"

class ClimateDatasetV2C(ClimateDatasetV2A):
    DATE_PATH = "split_datesC.txt"

class ClimateDatasetV2D(ClimateDatasetV2A):
    STATION_PATH = "station_split.txt"

    def __init__(self, root_dir="../climate_new/", transform='MeanStdNormalize', split='train', stats=None, 
                 seqlen=365, testNum=None, x_feature=None, y_feature=None, exclude=0, retDate=False, minmax_feature=None):
        """
        Args:
            root_dir (str): 
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        if split not in ['train', 'test', 'testsubset']:
            raise ValueError('split should be one of train, test, testsubset')
        self.split = split
        self.seqlen = seqlen
        
        self.retDate = retDate
        self.x_feature = x_feature
        self.y_feature = y_feature
        
        df_station = pd.read_csv(os.path.join(root_dir, "static_filtered.csv"), dtype={'STAID': str})
        df_station['STAID'] = df_station['STAID'].apply(lambda x: x.zfill(8))

        ### 加
        self.station_colnames = df_station.columns[1:]   # 记住静态特征列名（不含 STAID）
        ####################

        # df_station = df_station.drop(columns=['SNOW_PCT_PRECIP'])
        # df_station.drop(columns=['HGAC'], inplace=True)
        # if split is 'test', randomly select 50 stations for testing
        if split == 'test' and testNum:
            df_station = df_station.sample(n=testNum, random_state=42)
        elif split == 'all' and testNum:
            df_station = df_station.sample(n=testNum, random_state=42)
        # else:
        #     df_station = df_station.sample(n=100, random_state=42)

        if exclude:
            import json
            with open("../group.json", 'r') as f:
                group = json.load(f)
            if isinstance(exclude, int):
                exclude_features = group[str(exclude)]
            else:
                exclude_features = []
                for e in exclude:
                    exclude_features += group[str(e)]
        else:
            exclude_features = []
        if x_feature:
            x_feature = [col for col in x_feature if col not in exclude_features]
            self.x_feature = x_feature
        
        temporal = []
        targets = []
        self.sequences = []
        self.labels = []
        self.sample_info = []
        self.station_ids = []
        
        self.dates = list(pd.date_range(start='1982-01-01', end='2018-12-31', freq='D'))
        
        with open(os.path.join(root_dir, self.STATION_PATH), "r") as f:
            for line in f.readlines():
                staid, mode = line.split(" ")
                staid = staid.strip()
                mode = mode.strip()
                # if split == 'all':
                #     self.sample_info.extend([(staid, pd.Timestamp(date.strip())) for date in self.dates])

                if mode == split:
                    # self.sample_info.extend([(staid, pd.Timestamp(date.strip())) for date in self.dates])
                    self.station_ids.append(staid)
                if split == "testsubset" and mode == "test":
                    self.station_ids.append(staid)
                # if mode == 'test':
                #     test_dates.append(staid)

        print(len(self.sample_info))
        self.station_temporal = {}
        self.station_feature = {}
        
        for staid in tqdm(self.station_ids):
            staid_str = staid.zfill(8)
            df = pd.read_csv(os.path.join(root_dir, f"{staid_str}.csv"))
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

            dates = df['Date'].values
            if split == 'testsubset':
                for date in dates:
                    if int(date.split("-")[0]) in self.test_year:
                        self.sample_info.extend([(staid, pd.Timestamp(date.strip()))])
            else:
                self.sample_info.extend([(staid, pd.Timestamp(date.strip())) for date in dates])
            
            if split == "train":
                station = df_station.loc[df_station['STAID'] == staid]
                if exclude:
                    station = station.drop(columns=[col for col in exclude_features if col in station.columns])
                station = station.iloc[:, 1:].values
                self.station_feature[staid_str] = station

                ### 加
                # station 是 numpy，形状 [1, n_static]；转成 dict 方便取值
                station_cols = self.station_colnames.tolist()
                station_vals = self.station_feature[staid_str][0]  # 该站静态特征的一行
                static_map = {col: station_vals[i] for i, col in enumerate(station_cols)}
                ####################
                df_train = df[df['Date'].isin(self.dates)]
                if x_feature:
                    temporal_data = df.loc[:, x_feature].values
                else:
                    temporal_data = df.iloc[:, 21:].values

                temporal.append(temporal_data)
                
                if y_feature:
                    # 加：找出 y_feature 里不在 df_train 的列（这些应该都是静态列）
                    missing = [c for c in y_feature if c not in df_train.columns]
                    if len(missing) > 0:
                        # 用该站的静态值，把缺失列补到 df_train，每列值全相同
                        for c in missing:
                            if c not in static_map:
                                raise KeyError(f"Static feature '{c}' not found in static_filtered.csv")
                            df_train[c] = static_map[c]
                    target = df_train.loc[:, y_feature].values
                else:
                    target = df.iloc[:, 1:21].values
                targets.append(target)

                
                df['Date'] = pd.to_datetime(df['Date'])
                self.station_temporal[staid_str] = df
            
            else:
                station = df_station.loc[df_station['STAID'] == staid]
                if exclude:
                    station = station.drop(columns=[col for col in exclude_features if col in station.columns])
                   
                station = station.iloc[:, 1:].values
                self.station_feature[staid_str] = station

                df['Date'] = pd.to_datetime(df['Date'])
                self.station_temporal[staid_str] = df
            
        if minmax_feature:
            station_columns = df_station.columns
            if exclude:
                station_columns = station_columns.drop([col for col in exclude_features if col in station_columns])
            station_minmax_idx = station_columns.get_indexer(minmax_feature)
            # filter out -1 in idx
            station_minmax_idx = station_minmax_idx[station_minmax_idx != -1]
            # print(station.columns[station_minmax_idx]) # this yields ['LAT_GAGE', 'LNG_GAGE']
            station_minmax_idx -= 1

            if x_feature:
                temporal_minmax_idx = [x_feature.index(col) for col in minmax_feature if col in x_feature]
                # print(temporal_minmax_idx)
            else:
                col = df.columns[21:]
                temporal_minmax_idx = col.get_indexer(minmax_feature)
                temporal_minmax_idx = temporal_minmax_idx[temporal_minmax_idx != -1]
            
            if y_feature:
                self.y_minmax_idx = [y_feature.index(col) for col in minmax_feature if col in y_feature]
            else:
                col = df.columns[1:21]
                y_minmax_idx = col.get_indexer(minmax_feature)
                self.y_minmax_idx = y_minmax_idx[y_minmax_idx != -1]

        if split == 'train':
            self.temporal = from_numpy(np.concatenate(temporal, axis=0)).float()
            targets = np.concatenate(targets, axis=0)
            self.targets = from_numpy(targets).float()
            
            temp_stat = calculate_statistics(self.temporal)
            target_stat = calculate_statistics(self.targets)
            station_stat = self.get_station_stats()
        else:
            temp_stat = stats['temporal']
            target_stat = stats['target']
            station_stat = stats['station']
        
        if transform == "MeanStdNormalize":
            self.x_transform = MeanStdNormalize(
                temp_stat['mean'],
                temp_stat['variance']
            )
            self.station_transform = MeanStdNormalize(
                station_stat['mean'],
                station_stat['variance']
            )
        elif transform == "MinMaxNormalize":
            self.x_transform = MinMaxNormalize(
                temp_stat['min'],
                temp_stat['max']
            )
            self.station_transform = MinMaxNormalize(
                station_stat['min'],
                station_stat['max']
            )
        elif transform == "MinMaxNormalizeReal":
            self.x_transform = MinMaxNormalize(
                temp_stat['max'],
                temp_stat['min']
            )
            self.station_transform = MinMaxNormalize(
                station_stat['max'],
                station_stat['min']
            )
        elif transform == "LogMinMax":
            self.x_transform = LogMinMax(
                temp_stat,
                index=temporal_minmax_idx
            )
            self.station_transform = LogMinMax(
                station_stat,
                index=station_minmax_idx
            )
        self.y_transform = LogNormalize(
            target_stat,
            self.y_minmax_idx
        )
        # print("Applying minmax to y features: ", self.y_minmax_idx)
        return