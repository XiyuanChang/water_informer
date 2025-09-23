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
                # This is used for calculating statistics of train data (mean, std, etc.)
                df_train = df[df['Date'].isin(self.dates)]
                if x_feature:
                    temporal_data = df_train.loc[:, x_feature].values
                else:
                    temporal_data = df_train.iloc[:, 21:].values
                temporal.append(temporal_data)
                
                if y_feature:
                    target = df_train.loc[:, y_feature].values
                else:
                    target = df_train.iloc[:, 1:21].values
                targets.append(target)

                station = df_station.loc[df_station['STAID'] == staid]
                if exclude:
                    station = station.drop(columns=[col for col in exclude_features if col in station.columns])
                station = station.iloc[:, 1:].values
                self.station_feature[staid_str] = station

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
        
        if self.x_feature:
            x_feature_idx = np.array([df.columns.get_loc(col) for col in self.x_feature])
        if self.y_feature:
            y_feature_idx = np.array([df.columns.get_loc(col) for col in self.y_feature])

        # Find data from the required range
        start_date = current_date - pd.Timedelta(days=self.seqlen - 1)
        mask = (df['Date'] >= start_date) & (df['Date'] <= current_date)
        data = df.loc[mask].copy()

        # Convert DataFrame to NumPy array
        data_array = data.iloc[:, 1:].values

        # Handle case where the sequence is shorter than expected
        current_length = data_array.shape[0]
        if current_length < self.seqlen:
            pad_size = self.seqlen - current_length
            # Create padding with NaN for missing data points
            pad = np.full((pad_size, data_array.shape[1]), np.nan)
            # Concatenate pad and data
            data_array = np.vstack((pad, data_array))

        # Split into x and y components and convert to appropriate tensor type
        if self.x_feature:
            x = from_numpy(data_array[:, x_feature_idx-1]).float()
        else:
            x = from_numpy(data_array[:, 20:]).float()
        if self.y_feature:
            y = from_numpy(data_array[:, y_feature_idx-1]).float()
        else:
            y = from_numpy(data_array[:, :20]).float()

        station = self.station_feature[staid_str]
        station = from_numpy(np.repeat(station, x.shape[0], axis=0)).float()
        station = self.station_transform(station)
        # if station.isnan().sum() != 0:
        #     print(staid_str)
        #     print(station.isnan().sum())
        # assert station.isnan().sum() == 0
        x = self.x_transform(x)
        y = self.y_transform(y)
        
        x = torch.cat((x, station), dim=1)

        x[torch.isnan(x)] = -1
        # y[torch.isnan(y)] = -1

        if self.split == "train":
            return x, y
        elif self.split == "all":
            return staid_str, current_date.strftime('%Y-%m-%d'), x, y
        else:
            if self.retDate:
                return staid_str, current_date.strftime('%Y-%m-%d'), x, y
            return staid_str, x, y
        # else:
            staid, current_date = self.sample_info[idx]
            
            # staid = list(self.station_temporal.keys())[idx]
            staid_str = staid.zfill(8)
            df = self.station_temporal[staid_str]
            full_data = df.iloc[:, 1:].values
            
            station = self.station_feature[staid_str]

            for current_date in self.dates:
                current_date = pd.Timestamp(current_date)
                # find the index where df['Date'] == current_date
                idx = df.index[df['Date'] == current_date].tolist()
                if len(idx) == 0:
                    continue
                idx = idx[0]
                # Find data from the required range
                data_array = full_data[max(0, idx - self.seqlen + 1):idx + 1]

                # start_date = current_date - pd.Timedelta(days=self.seqlen - 1)
                # mask = (df['Date'] >= start_date) & (df['Date'] <= current_date)
                # data = df.loc[mask].copy()

                # Convert DataFrame to NumPy array
                # data_array = data.iloc[:, 1:].values

                # Handle case where the sequence is shorter than expected
                current_length = data_array.shape[0]
                if current_length < self.seqlen:
                    pad_size = self.seqlen - current_length
                    # Create padding with NaN for missing data points
                    pad = np.full((pad_size, data_array.shape[1]), np.nan)
                    # Concatenate pad and data
                    data_array = np.vstack((pad, data_array))

                # Split into x and y components and convert to appropriate tensor type
                x = from_numpy(data_array[:, 20:]).float()
                y = from_numpy(data_array[:, :20]).float()
  
                # Handle case where the sequence is shorter than expected
                # if len(data) < self.seqlen:
                #     pad_size = self.seqlen - len(data)
                #     pad = pd.DataFrame(None, index=pd.date_range(start_date, periods=pad_size), columns=df.columns)
                #     data = pd.concat([pad, data], ignore_index=True)
                
                # x = from_numpy(data.iloc[:, 21:].values).float()
                # y = from_numpy(data.iloc[:, 1:21].values).float()

                x = self.x_transform(x)
                y = self.y_transform(y)
                
                station_concat = from_numpy(np.repeat(station, x.shape[0], axis=0)).float()
                station_concat = self.station_transform(station_concat)
                x = torch.cat((x, station_concat), dim=1)

                x[torch.isnan(x)] = -1
                y[torch.isnan(y)] = -1

                final_x.append(x)
                final_y.append(y)
            
            return torch.stack(final_x), torch.stack(final_y)


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
                if x_feature:
                    temporal_data = df.loc[:, x_feature].values
                else:
                    temporal_data = df.iloc[:, 21:].values

                temporal.append(temporal_data)
                
                if y_feature:
                    target = df.loc[:, y_feature].values
                else:
                    target = df.iloc[:, 1:21].values
                targets.append(target)

                station = df_station.loc[df_station['STAID'] == staid]
                if exclude:
                    station = station.drop(columns=[col for col in exclude_features if col in station.columns])
                station = station.iloc[:, 1:].values
                self.station_feature[staid_str] = station

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