import os
import pandas as pd
from torch.utils.data import Dataset
from torch import from_numpy
from mytransform import MeanStdNormalize, MinMaxNormalize, MeanStdDeNormalize, MinMaxDeNormalize, LogNormalize, LogDeNormalize, LogMinMax, CustomQuantileTransformer
# from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pickle
import torch

class ClimateDataset(Dataset):
    def __init__(self, root_dir="Maumee DL/", branch_transform=None, trunk_transform=None, target_transform=None, split='train'):
        """
        Args:
            root_dir (str): 
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        if split not in ['train', 'val', 'test', 'all']:
            raise ValueError('split should be one of train, val, test, all')
        df_station = pd.read_csv(os.path.join(root_dir, "CrossSectionalData.csv"))
        temporal = []
        targets = []
        stations = []

        split_path = os.path.join(root_dir, 'train_test_split.csv')
        with open(split_path, "r") as f:
            for line in f.readlines():
                idx, mode = line.split(",")
                mode = mode.strip()
                idx = int(idx)
                
                if mode == split or split == 'all':
                    df = pd.read_csv(os.path.join(root_dir, f"{idx}_climate_sediment.csv"))
                    temporal_data = df.iloc[:, 1:7].values
                    temporal.append(temporal_data)

                    target = df.iloc[:, 7].values
                    targets.append(target)
                    
                    station = df_station.iloc[idx-1, 1:].values
                    station = np.repeat(station[np.newaxis, :], temporal_data.shape[0], axis=0)
                    assert station.shape[0] == target.shape[0]
                    stations.append(station)
        
        self.stations = from_numpy(np.concatenate(stations, axis=0)).float()
        self.targets = from_numpy(np.concatenate(targets, axis=0)).float()*1000
        self.temporal = from_numpy(np.concatenate(temporal, axis=0)).float()
        self.b_transforms = branch_transform
        self.t_transforms = trunk_transform
        self.target_transform = target_transform
    
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
        target = self.targets[idx].reshape([1,])

        if self.b_transforms:
            temporal = self.b_transforms(temporal)
        if self.t_transforms:
            station = self.t_transforms(station)
        if self.target_transform:
            target = self.target_transform(target)
        return temporal, station, target


class ClimateDatasetTime(ClimateDataset):
    def __init__(self, root_dir, branch_transform=None, trunk_transform=None, target_transform=None, split='train'):

        data_path = os.path.join(root_dir, 'train_test_data.pkl')
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        if split == 'train':
            self.stations = data["train_stations"]
            self.temporal = data["train_temporal"]
            self.targets = data["train_targets"]
        elif split == 'test':
            self.stations = data["test_stations"]
            self.temporal = data["test_temporal"]
            self.targets = data["test_targets"]
        elif split == 'all':
            self.stations = torch.cat([data["train_stations"], data["test_stations"]], dim=0)
            self.temporal = torch.cat([data["train_temporal"], data["test_temporal"]], dim=0)
            self.targets = torch.cat([data["train_targets"], data["test_targets"]], dim=0)
                    
        self.b_transforms = branch_transform
        self.t_transforms = trunk_transform
        self.target_transform = target_transform


class ClimateDatasetV2(Dataset):
    def __init__(self, root_dir="../climate_washed/", branch_transform=None, trunk_transform=None, target_transform=None, split='train'):
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

    def __init__(self, root_dir="../climate_washed/", transform='MeanStdNormalize', split='train', 
                 stats=None, x_feature=None, y_feature=None, exclude=0, location_static=False,
                 minmax_feature=None):
        """
        Args:
            root_dir (str): 
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        if split not in ['train', 'test', 'all']:
            raise ValueError('split should be one of train, val, test, all')
        self.split = split
        
        df_station = pd.read_csv(os.path.join(root_dir, "static_filtered.csv"), dtype={'STAID': str})
        # df_station = df_station.drop(columns=['SNOW_PCT_PRECIP'])
        
        temporal = []
        targets = []
        stations = []
        
        dates = []
        
        with open(os.path.join(root_dir, self.DATE_PATH), "r") as f:
            for line in f.readlines():
                date, mode = line.split(" ")
                date = date.strip()
                mode = mode.strip()
                
                if mode == split:
                    dates.append(date)
        if exclude:
            import json
            with open("../group.json") as f:
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
            if exclude:
                x_feature = [f for f in x_feature if f not in exclude_features]
        else:
            df = pd.read_csv(os.path.join(root_dir, "01054200.csv"))
            x_feature = df.columns[21:]

        for staid in df_station['STAID']:
            if len(staid) < 8:
                staid_str = '0' * (8 - len(staid)) + staid
            else:
                staid_str = staid
            df = pd.read_csv(os.path.join(root_dir, f"{staid_str}.csv"))
            df = df[df['Date'].isin(dates)]
            if df.shape[0] == 0:
                continue
            # print(staid_str, df.shape[0])
            
            # if x_feature:
            #     if exclude:
            #         x_feature = [f for f in x_feature if f not in exclude_features]
            #     temporal_data = df[x_feature].values
            #     print(len(x_feature))
            # else:
            #     temporal_data = df.iloc[:, 21:].values
            temporal_data = df[x_feature].values
            
            if y_feature:
                # print(df.columns)
                target = df[y_feature].values
                mask = ~(np.isnan(target).all(axis=1))
                if mask.sum() == 0:
                    continue
                target = target[mask]
                temporal_data = temporal_data[mask]
            else:
                target = df.iloc[:, 1:21].values

            station = df_station.loc[df_station['STAID'] == staid]
            if exclude:
                drop_columns = [f for f in exclude_features if f in station.columns]
                if location_static:
                    # if "LAT_GAGE,LNG_GAGE" column is in drop_columns, we should keep it
                    if 'LAT_GAGE' in drop_columns:
                        drop_columns.remove('LAT_GAGE')
                    if 'LNG_GAGE' in drop_columns:
                        drop_columns.remove('LNG_GAGE')
                station = station.drop(columns=drop_columns)
            assert isinstance(station, pd.DataFrame)

            if location_static:
                # preserve only the "LAT_GAGE,LNG_GAGE" column in station, the rest of station is concatenated with temporal
                station_rest = station.drop(columns=['STAID', 'LAT_GAGE', 'LNG_GAGE'])
                station_columns = station.columns
                station_rest = station_rest.values
                station_rest = np.repeat(station_rest, temporal_data.shape[0], axis=0)

                negative_columns = (station_rest < 0).any(axis=0)

                temporal_date = df[['datenum', 'sinT', 'cosT']].values

                station = station[['LAT_GAGE', 'LNG_GAGE']].values

                station = np.repeat(station, temporal_data.shape[0], axis=0)
                station = np.concatenate([temporal_date, station], axis=1)
                # print(df.columns)

                x_feature_no_date = [f for f in x_feature if f not in ['datenum', 'sinT', 'cosT']]
                temporal_rest = df[x_feature_no_date].values
                # temporal_data = np.concatenate([temporal_data, station_rest], axis=1)
                temporal_data = np.concatenate([temporal_rest, station_rest], axis=1)
                # print(temporal_data.shape)

                if minmax_feature:
                    temporal_columns = x_feature_no_date + station_columns.drop(['STAID', 'LAT_GAGE', 'LNG_GAGE']).to_list()
                    temporal_minmax_idx = [temporal_columns.index(f) for f in minmax_feature if f in temporal_columns]

                    station_columns = ['datenum', 'sinT', 'cosT', 'LAT_GAGE', 'LNG_GAGE']
                    station_minmax_idx = [station_columns.index(f) for f in minmax_feature if f in station_columns]

                    target_columns = y_feature if y_feature else df.columns[1:21]
                    self.target_minmax_idx = [target_columns.index(f) for f in minmax_feature if f in target_columns]
                    # print("Using minmax index for y: ", self.target_minmax_idx)

            else:
                station = station.iloc[:, 1:].values
                station = np.repeat(station, temporal_data.shape[0], axis=0)

            assert station.shape[0] == target.shape[0]
            temporal.append(temporal_data)
            targets.append(target)
            stations.append(station)

        if split == 'train':
            self.stations = from_numpy(np.concatenate(stations, axis=0)).float()
            self.temporal = from_numpy(np.concatenate(temporal, axis=0)).float()
            targets = np.concatenate(targets, axis=0)
            self.targets = from_numpy(targets).float()
            print("Station shape: ", self.stations.shape)
            print("Temporal shape: ", self.temporal.shape)
            print(targets.shape)

            # mask = ~np.isnan(targets)
            # tensor = from_numpy(targets[mask]).float()
            # tensor_nan = torch.full(targets.shape, torch.nan, dtype=torch.float32)
            # mask_flat = mask.flatten()
            # tensor_nan_flat = tensor_nan.flatten()
            # tensor_nan_flat[mask_flat] = tensor
            # self.targets = tensor_nan_flat.reshape(targets.shape)
        else:
            self.stations = stations
            self.temporal = temporal
            self.targets = targets
            print(from_numpy(np.concatenate(self.targets, axis=0)).float().shape)

        if stats:
            temp_stat = stats['temporal']
            target_stat = stats['target']
            station_stat = stats['station']
        else:
            temp_stat = calculate_statistics(self.temporal)
            target_stat = calculate_statistics(self.targets)
            station_stat = calculate_statistics(self.stations)
            # print("y variance: ", target_stat['variance'])
            
            self.transform = transform
        
        if transform == "MeanStdNormalize":
            self.b_transforms = MeanStdNormalize(
                temp_stat['mean'],
                temp_stat['variance']
            )
            self.t_transforms = MeanStdNormalize(
                station_stat['mean'],
                station_stat['variance']
            )
        elif transform == "MinMaxNormalize":
            self.b_transforms = MinMaxNormalize(
                temp_stat['min'],
                temp_stat['max']
            )
            self.t_transforms = MinMaxNormalize(
                station_stat['min'],
                station_stat['max']
            )
        elif transform == "LogMinMax":
            if not location_static:
                raise ValueError("location_static must be True for MinMaxNormalizeV2")
            self.b_transforms = LogMinMax(
                temp_stat,
                index=temporal_minmax_idx
            )
            self.t_transforms = LogMinMax(
                station_stat,
                index=station_minmax_idx
            )
        elif transform == "MinMaxNormalizeReal":
            self.b_transforms = MinMaxNormalize(
                temp_stat['max'],
                temp_stat['min']
            )
            self.t_transforms = MinMaxNormalize(
                station_stat['max'],
                station_stat['min']
            )

        self.target_transform = LogNormalize(
            target_stat,
            index=self.target_minmax_idx
        )

    def __len__(self):
        if isinstance(self.targets, list):
            return len(self.targets)
        else:
            return self.targets.shape[0]

    def get_stats(self):
        return {
            'temporal': calculate_statistics(self.temporal),
            'station': calculate_statistics(self.stations),
            'target': calculate_statistics(self.targets)
        }

    def get_target_detransform(self):
        target_stat = calculate_statistics(self.targets)
        return LogDeNormalize(
            target_stat,
            self.target_minmax_idx
        )
        if self.transform == "MeanStdNormalize":
            return MeanStdDeNormalize(
                target_stat['mean'],
                target_stat['variance']
            )
        else:
            return MinMaxDeNormalize(
                target_stat['min'],
                target_stat['max']
            )

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
        if self.split == 'test':
            # import pdb; pdb.set_trace()
            station = from_numpy(station).float()
            temporal = from_numpy(temporal).float()
            
            target = from_numpy(target).float()
            # mask = ~np.isnan(target)
            # tensor = from_numpy(target[mask]).float()
            # tensor_nan = torch.full(target.shape, torch.nan, dtype=torch.float32)
            # mask_flat = mask.flatten()
            # tensor_nan_flat = tensor_nan.flatten()
            # tensor_nan_flat[mask_flat] = tensor
            # target = tensor_nan_flat.reshape(target.shape)
        
        temporal = self.b_transforms(temporal)
        station = self.t_transforms(station)
        target = self.target_transform(target)
        if torch.isnan(temporal).any():
        #     # find out which index has nan
            print("Temporal has nan, idx: ", idx)
            print(torch.isnan((temporal)).nonzero())
            print(self.temporal[idx][torch.isnan((temporal)).nonzero()])
        return temporal, station, target

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
    variance = torch.sqrt(variance)
    # min, max
    max = masked_tensor.max(dim=0)[0]
    min = masked_tensor.min(dim=0)[0]
    
    y = tensor.float()
    y_log = torch.log(y.clamp(min=1e-9))
    y_log_add = torch.log(y+1e-4)
    log_max = nanmax(y_log_add, dim=0)[0]
    log_min = nanmin(y_log_add, dim=0)[0]
    log_mean = torch.nanmean(y_log_add, dim=0)
    log_var = nanvariance(y_log_add, dim=0)

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

def nanvariance(t, dim=None):
    """
    Calculate variance of a tensor along a specified dimension, ignoring NaNs.
    
    Arguments:
    t (torch.Tensor): Input tensor. Can be 1D or 2D. Can contain NaNs.
    dim (int, optional): The dimension along which to calculate variance. Default is None (apply on entire tensor).
    unbiased (bool): Whether to use the unbiased estimator (N-1) in the variance calculation. Default is True.
    
    Returns:
    torch.Tensor: Variance of the tensor, ignoring NaN values.
    """
    # Mask for non-NaN values
    mask = ~torch.isnan(t)
    
    # Replace NaNs with zeros (this is safe since we are masking them out later)
    t_nan_replaced = torch.where(mask, t, torch.tensor(0.0, device=t.device, dtype=t.dtype))
    
    # Count the number of non-NaN values along the specified dimension
    num_non_nan = mask.sum(dim=dim, keepdim=True)
    
    # Calculate mean ignoring NaNs
    mean_non_nan = t_nan_replaced.sum(dim=dim, keepdim=True) / num_non_nan
    
    # Calculate squared differences from the mean, ignoring NaNs
    squared_diff = torch.where(mask, (t - mean_non_nan) ** 2, torch.tensor(0, device=t.device, dtype=t.dtype))
    
    # Sum of squared differences, ignoring NaNs
    sum_squared_diff = squared_diff.sum(dim=dim, keepdim=False)
    dof = num_non_nan
    
    # Avoid division by zero (in case all values are NaN in a slice)
    dof = dof.clamp(min=1)
    
    # Calculate variance
    variance = sum_squared_diff / dof.squeeze(dim)
    
    return variance

class ClimateDatasetV2B(ClimateDatasetV2A):
    DATE_PATH = "split_datesB.txt"

class ClimateDatasetV2C(ClimateDatasetV2A):
    DATE_PATH = "split_datesC.txt"


class ClimateDatasetV2D(ClimateDatasetV2A):
    STATION_PATH = "station_split3.txt"

    def __init__(self, root_dir="../climate_washed/", transform='MeanStdNormalize', split='train', 
                 stats=None, x_feature=None, y_feature=None, exclude=0, location_static=False):
        if split not in ['train', 'test', 'all']:
            raise ValueError('split should be one of train, val, test, all')
        self.split = split

        df_station = pd.read_csv(os.path.join(root_dir, "static_filtered.csv"), dtype={'STAID': str})
        
        temporal = []
        targets = []
        stations = []

        if exclude:
            import json
            with open("../group.json") as f:
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
            if exclude:
                x_feature = [f for f in x_feature if f not in exclude_features]
        else:
            df = pd.read_csv(os.path.join(root_dir, "01054200.csv"))
            x_feature = df.columns[21:]
        
        station_ids = []
        with open(os.path.join(root_dir, self.STATION_PATH), "r") as f:
            for l in f.readlines():
                idx, mode = l.split(" ")
                mode = mode.strip()
                
                if mode == split:
                    station_ids.append(idx)
        
        for staid in station_ids:
            staid_str = staid.zfill(8)
            df = pd.read_csv(os.path.join(root_dir, f"{staid_str}.csv"))
            if df.shape[0] == 0:
                continue

            temporal_data = df[x_feature].values

            if y_feature:
                # print(df.columns)
                target = df[y_feature].values
                mask = ~(np.isnan(target).all(axis=1))
                if mask.sum() == 0:
                    continue
                target = target[mask]
                temporal_data = temporal_data[mask]
            else:
                target = df.iloc[:, 1:21].values

            station = df_station.loc[df_station['STAID'] == staid.lstrip('0')]
            if exclude:
                drop_columns = [f for f in exclude_features if f in station.columns]
                if location_static:
                    # if "LAT_GAGE,LNG_GAGE" column is in drop_columns, we should keep it
                    if 'LAT_GAGE' in drop_columns:
                        drop_columns.remove('LAT_GAGE')
                    if 'LNG_GAGE' in drop_columns:
                        drop_columns.remove('LNG_GAGE')
                station = station.drop(columns=drop_columns)
            assert isinstance(station, pd.DataFrame)

            if location_static:
                # preserve only the "LAT_GAGE,LNG_GAGE" column in station, the rest of station is concatenated with temporal
                station_rest = station.drop(columns=['STAID', 'LAT_GAGE', 'LNG_GAGE'])
                station_rest = station_rest.values
                station_rest = np.repeat(station_rest, temporal_data.shape[0], axis=0)

                temporal_date = df[['datenum', 'sinT', 'cosT']].values

                station = station[['LAT_GAGE', 'LNG_GAGE']].values

                station = np.repeat(station, temporal_data.shape[0], axis=0)
                station = np.concatenate([temporal_date, station], axis=1)
                # print(df.columns)

                x_feature_no_date = [f for f in x_feature if f not in ['datenum', 'sinT', 'cosT']]
                temporal_rest = df[x_feature_no_date].values
                # temporal_data = np.concatenate([temporal_data, station_rest], axis=1)
                temporal_data = np.concatenate([temporal_rest, station_rest], axis=1)
                # print(temporal_data.shape)

            else:
                station = station.iloc[:, 1:].values
                station = np.repeat(station, temporal_data.shape[0], axis=0)
            
            # import pdb; pdb.set_trace()
            assert station.shape[0] == target.shape[0]
            temporal.append(temporal_data)
            targets.append(target)
            stations.append(station)

        if split == 'train':
            self.stations = from_numpy(np.concatenate(stations, axis=0)).float()
            self.temporal = from_numpy(np.concatenate(temporal, axis=0)).float()
            targets = np.concatenate(targets, axis=0)
            self.targets = from_numpy(targets).float()
            print("Station shape: ", self.stations.shape)
            print("Temporal shape: ", self.temporal.shape)
            print(targets.shape)

            # mask = ~np.isnan(targets)
            # tensor = from_numpy(targets[mask]).float()
            # tensor_nan = torch.full(targets.shape, torch.nan, dtype=torch.float32)
            # mask_flat = mask.flatten()
            # tensor_nan_flat = tensor_nan.flatten()
            # tensor_nan_flat[mask_flat] = tensor
            # self.targets = tensor_nan_flat.reshape(targets.shape)
        else:
            self.stations = stations
            self.temporal = temporal
            self.targets = targets
            print(from_numpy(np.concatenate(self.targets, axis=0)).float().shape)

        if stats:
            temp_stat = stats['temporal']
            target_stat = stats['target']
            station_stat = stats['station']
        else:
            temp_stat = calculate_statistics(self.temporal)
            target_stat = calculate_statistics(self.targets)
            station_stat = calculate_statistics(self.stations)
            # print("y variance: ", target_stat['variance'])
            
            self.transform = transform
        
        if transform == "MeanStdNormalize":
            self.b_transforms = MeanStdNormalize(
                temp_stat['mean'],
                temp_stat['variance']
            )
            self.t_transforms = MeanStdNormalize(
                station_stat['mean'],
                station_stat['variance']
            )
            self.target_transform = MeanStdNormalize(
                target_stat['mean'],
                target_stat['variance']
            )
        elif transform == "MinMaxNormalize":
            self.b_transforms = MinMaxNormalize(
                temp_stat['min'],
                temp_stat['max']
            )
            self.t_transforms = MinMaxNormalize(
                station_stat['min'],
                station_stat['max']
            )
            self.target_transform = MinMaxNormalize(
                target_stat['min'],
                target_stat['max']
            )
        elif transform == "LogMinMax":
            if not location_static:
                raise ValueError("location_static must be True for MinMaxNormalizeV2")
            self.b_transforms = LogMinMax(
                temp_stat['logmin'],
                temp_stat['logmax'],
                temp_stat['min'],
                temp_stat['max'],
                index=-1
            )
            self.t_transforms = LogMinMax(
                station_stat['logmin'],
                station_stat['logmax'],
                station_stat['min'],
                station_stat['max'],
                index=3
            )
            import pdb; pdb.set_trace()
        elif transform == "MinMaxNormalizeReal":
            self.b_transforms = MinMaxNormalize(
                temp_stat['max'],
                temp_stat['min']
            )
            self.t_transforms = MinMaxNormalize(
                station_stat['max'],
                station_stat['min']
            )
        self.target_transform = LogNormalize(
            target_stat['10th'],
            target_stat['90th']
        )
