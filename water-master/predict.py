import os
import pandas as pd
from mytransform import MeanStdNormalize, MinMaxNormalize, MeanStdDeNormalize, MinMaxDeNormalize, LogDeNormalize, LogMinMax
from model.metric import _pbias, _r_squared, _kge, pbias, r_squared, kge, _r, _beta, _alpha, _nse
import torch
import numpy as np
import copy
from model import model as module_model
from tqdm import tqdm
from parse_config import ConfigParser
import argparse
import torch.nn as nn
import dataset
import json

def predict_whole_dataset(model, state_dicts, stats, valid_threshold=51, normalize='MeanStdNormalize',
                          split='split_datesA.txt', root="../climate_new", output_dir="output",
                          x_feature=None, y_feature=None, exclude=None, location_static=False, minmax_feature=None):
    """0：有效的用于train的y 1：有效的用于test的y 2：x齐全可以预测出来且不是0和1 3：x不全 4:有效的用于test的y,但是y数量太少"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metric_dir = os.path.join(output_dir, "metric")
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    if split.startswith("station"):
        # setting D: 80-20 split station
        predict_station(model, state_dicts, stats, valid_threshold=valid_threshold, normalize=normalize,
                        split=split, root=root, output_dir=output_dir, x_feature=x_feature, y_feature=y_feature,
                        exclude=exclude, location_static=location_static)
        return

    with open("../group.json") as f:
        group = json.load(f)
        if exclude:
            if isinstance(exclude, int):
                exclude_features = group[str(exclude)]
            else:
                exclude_features = []
                for e in exclude:
                    exclude_features += group[str(e)]

    model_list = []
    for state in state_dicts:
        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict(state)
        model_list.append(copy.deepcopy(model_copy))
    
    with open(os.path.join(root, "static_filtered.csv")) as f:
        df_station = pd.read_csv(f, dtype={'STAID': str})
    # df_station = df_station.drop(columns=['SNOW_PCT_PRECIP'])

    temp_stat = stats['temporal']
    target_stat = stats['target']
    station_stat = stats['station']
    if normalize == "MeanStdNormalize":
        temporal_transform = MeanStdNormalize(
            temp_stat['mean'],
            temp_stat['variance']
        )
        station_transform = MeanStdNormalize(
            station_stat['mean'],
            station_stat['variance']
        )
        target_detransform = MeanStdDeNormalize(
            target_stat['mean'],
            target_stat['variance']
        )
    elif normalize == "MinMaxNormalize":
        temporal_transform = MinMaxNormalize(
            temp_stat['min'],
            temp_stat['max']
        )
        station_transform = MinMaxNormalize(
            station_stat['min'],
            station_stat['max']
        )
        target_detransform = MinMaxDeNormalize(
            target_stat['min'],
            target_stat['max']
        )
    elif normalize == "LogMinMax":
        pass # will deal with it later
    elif normalize == "MinMaxNormalizeReal":
        temporal_transform = MinMaxNormalize(
            temp_stat['max'],
            temp_stat['min']
        )
        station_transform = MinMaxNormalize(
            station_stat['max'],
            station_stat['min']
        )
    target_columns = y_feature if y_feature else df.columns[1:21]
    target_minmax_index = [target_columns.index(f) for f in minmax_feature if f in target_columns]
    target_detransform = LogDeNormalize(
        target_stat,
        index=target_minmax_index
    )
    train_dates, test_dates = [], []
    with open(os.path.join(root, split), "r") as f:
        for line in f.readlines():
            date, mode = line.split(" ")
            date = date.strip()
            mode = mode.strip()
            
            if mode == 'train':
                train_dates.append(date)
            elif mode == 'test':
                test_dates.append(date)

    # metric bookkeeping
    col = ['staid']
    df = pd.read_csv(os.path.join(root, f"01054200.csv"))
    if y_feature:
        for name in y_feature:
            col.append(f"{name}_kge")
            col.append(f"{name}_pbias")
            col.append(f"{name}_r2")
            # col.append(f"{name}_num")
            # col.append(f"{name}_r")
            # col.append(f"{name}_beta")
            # col.append(f"{name}_alpha")
            # col.append(f"{name}_nse")
    else:
        for name in df.columns[1:21]:
            col.append(f"{name}_kge")
            col.append(f"{name}_pbias")
            col.append(f"{name}_r2")
            # col.append(f"{name}_num")
            # col.append(f"{name}_r")
            # col.append(f"{name}_beta")
            # col.append(f"{name}_alpha")
            # col.append(f"{name}_nse")
    metric_df = pd.DataFrame(columns=col)
    metric_df['staid'] = df_station['STAID']

    if not x_feature:
        x_feature = df.columns[21:]
    if exclude:
        x_feature = [x for x in x_feature if x not in exclude_features]

    for station_index, staid in enumerate(tqdm(df_station['STAID'])):
        # print("Saving results for station: ", staid)
        if len(staid) < 8:
            staid_str = '0' * (8 - len(staid)) + staid
        else:
            staid_str = staid
        df = pd.read_csv(os.path.join(root, f"{staid_str}.csv"))
        
        if y_feature:
            output_df = df[y_feature].copy()
            col_names = y_feature
        else:
            output_df = df.iloc[:, :21].copy()
            col_names = df.columns[1:21]

        for name in col_names:
            output_df[f"{name}_pred"] = np.nan
        for name in col_names:
            output_df[f"{name}_type"] = 3 # 默认x不全

        if x_feature:
            temporal = df.loc[:, x_feature].values
        else:
            temporal = df.iloc[:, 21:].values

        if y_feature:
            target = df.loc[:, y_feature].values
            # mask = ~(np.isnan(target).all(axis=1))
            # if mask.sum() == 0:
            #     pass # TODO
            # target = target[mask]
            # temporal = temporal[mask]
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

        if location_static:
            # preserve only the "LAT_GAGE,LNG_GAGE" column in station, the rest of station is concatenated with temporal
            station_rest = station.drop(columns=['STAID', 'LAT_GAGE', 'LNG_GAGE'])
            station_columns = station.columns
            station_rest = station_rest.values
            station_rest = np.repeat(station_rest, temporal.shape[0], axis=0)

            station = station[['LAT_GAGE', 'LNG_GAGE']].values
            station = np.repeat(station, temporal.shape[0], axis=0)

            temporal_date = df[['datenum', 'sinT', 'cosT']].values
            station = np.concatenate([temporal_date, station], axis=1)

            temporal = np.concatenate([temporal, station_rest], axis=1)

            x_feature_no_date = [f for f in x_feature if f not in ['datenum', 'sinT', 'cosT']]
            temporal_rest = df[x_feature_no_date].values
            temporal = np.concatenate([temporal_rest, station_rest], axis=1)

            if minmax_feature and normalize == "LogMinMax":
                temporal_columns = x_feature_no_date + station_columns.drop(['STAID', 'LAT_GAGE', 'LNG_GAGE']).to_list()
                temporal_minmax_idx = [temporal_columns.index(f) for f in minmax_feature if f in temporal_columns]
                station_columns = ['datenum', 'sinT', 'cosT', 'LAT_GAGE', 'LNG_GAGE']
                station_minmax_idx = [station_columns.index(f) for f in minmax_feature if f in station_columns]
                temporal_transform = LogMinMax(
                    temp_stat,
                    index=temporal_minmax_idx
                )
                station_transform = LogMinMax(
                    station_stat,
                    index=station_minmax_idx
                )
        else:
            station = station.iloc[:, 1:].values
        station = torch.from_numpy(station).float()

        for idx, model in enumerate(model_list):
            model.eval()
            model = model.cuda()
            with torch.no_grad():
                temporal_tensor = torch.from_numpy(temporal).float()
                temporal_tensor = temporal_transform(temporal_tensor)
                temporal_tensor = temporal_tensor.cuda()
                
                station_tensor = station.repeat(temporal_tensor.shape[0], 1)
                # station_tensor = np.repeat(station, temporal_tensor.shape[0], axis=0)
                station_tensor = station_transform(station)
                station_tensor = station_tensor.cuda()
                
                output = model(temporal_tensor, station_tensor)
                output = target_detransform(output)
                output = output.detach().cpu().numpy()

                for i, name in enumerate(col_names):
                    if i == idx:
                        output_df[f"{name}_pred"] = output[:, i]
                        
                        # 2：x齐全，可以预测出来且不是0和1
                        # if x_feature:
                        #     mask = df.loc[:, x_feature].notna().all(axis=1)
                        # else:
                        mask = df.iloc[:, 21:].notna().all(axis=1)
                        # mask &= df.iloc[:, 1:21].isna().all(axis=1)
                        output_df.loc[mask, f"{name}_type"] = 2

                        # 0：x齐全，有效的用于train的y
                        train_mask = df['Unnamed: 0'].isin(train_dates)
                        # train_mask &= df.iloc[:, int(i+1)].notna().any(axis=1)
                        train_mask &= df.loc[:,name].notna()
                        train_mask &= mask
                        output_df.loc[train_mask, f"{name}_type"] = 0

                        # 1：x齐全，有效的用于test的y
                        # 4: x齐全，有效的用于test的y，但是y数量太少
                        test_mask = df['Unnamed: 0'].isin(test_dates)
                        # test_mask &= df.iloc[:, int(i+1)].notna().any(axis=1)
                        test_mask &= df.loc[:, name].notna()
                        test_mask &= mask

                        if test_mask.sum() < valid_threshold:
                            output_df.loc[test_mask, f"{name}_type"] = 4
                        else:
                            output_df.loc[test_mask, f"{name}_type"] = 1
                        
                        # calculate metrics on test dataset
                        test_targets = df.loc[test_mask, name].values
                        test_mask = test_mask.to_numpy()
                        test_pred = output[test_mask, i]
                        if (~np.isnan(test_targets)).sum() < valid_threshold:
                            metric_df.loc[metric_df['staid'] == staid, f"{name}_num"] = (~np.isnan(test_targets)).sum()
                            continue
                        test_targets = torch.from_numpy(test_targets).float()
                        test_pred = torch.from_numpy(test_pred).float()
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_kge"] = _kge(test_pred, test_targets)
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_pbias"] = _pbias(test_pred, test_targets)
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_r2"] = _r_squared(test_pred, test_targets)
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_num"] = test_mask.sum()
                        # metric_df.loc[metric_df['staid'] == staid, f"{name}_r"] = _r(test_pred, test_targets)
                        # metric_df.loc[metric_df['staid'] == staid, f"{name}_beta"] = _beta(test_pred, test_targets)
                        # metric_df.loc[metric_df['staid'] == staid, f"{name}_alpha"] = _alpha(test_pred, test_targets)
                        # metric_df.loc[metric_df['staid'] == staid, f"{name}_nse"] = _nse(test_pred, test_targets)
        # if station_index == 10:
            # break
        output_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        with open(os.path.join(output_dir, f"{staid_str}.csv"), "w") as f:
            output_df.to_csv(f, index=False, na_rep='')
    print("Saving metrics to ", metric_dir, "/metric.csv")
    metric_df.to_csv(os.path.join(metric_dir, "metric.csv"), index=False, na_rep='')

def predict_station(model, state_dicts, stats, valid_threshold=51, normalize='MeanStdNormalize',
                          split='split_datesA.txt', root="../climate_new", output_dir="output",
                          x_feature=None, y_feature=None, exclude=None, location_static=False):
    
    with open("../group.json") as f:
        group = json.load(f)
        if exclude:
            if isinstance(exclude, int):
                exclude_features = group[str(exclude)]
            else:
                exclude_features = []
                for e in exclude:
                    exclude_features += group[str(e)]

    model_list = []
    for state in state_dicts:
        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict(state)
        model_list.append(copy.deepcopy(model_copy))
    
    with open(os.path.join(root, "static_filtered.csv")) as f:
        df_station = pd.read_csv(f, dtype={'STAID': str})
    
    temp_stat = stats['temporal']
    target_stat = stats['target']
    station_stat = stats['station']
    if normalize == "MeanStdNormalize":
        temporal_transform = MeanStdNormalize(
            temp_stat['mean'],
            temp_stat['variance']
        )
        station_transform = MeanStdNormalize(
            station_stat['mean'],
            station_stat['variance']
        )
        target_detransform = MeanStdDeNormalize(
            target_stat['mean'],
            target_stat['variance']
        )
    elif normalize == "MinMaxNormalize":
        temporal_transform = MinMaxNormalize(
            temp_stat['min'],
            temp_stat['max']
        )
        station_transform = MinMaxNormalize(
            station_stat['min'],
            station_stat['max']
        )
        target_detransform = MinMaxDeNormalize(
            target_stat['min'],
            target_stat['max']
        )
    target_detransform = LogDeNormalize(
        target_stat
    )
    raise NotImplementedError("Not implemented target transform in D setting yet")

    train_station, test_station = [], []
    print("Using split: ", split)
    with open(os.path.join(root, split), "r") as f:
        for line in f.readlines():
            station, mode = line.split(" ")
            station = station.strip()
            mode = mode.strip()
            
            if mode == 'train':
                train_station.append(station)
            elif mode == 'test':
                test_station.append(station)
    
    # metric bookkeeping
    col = ['staid']
    df = pd.read_csv(os.path.join(root, f"01054200.csv"))
    if y_feature:
        for name in y_feature:
            col.append(f"{name}_kge")
            col.append(f"{name}_pbias")
            col.append(f"{name}_r2")
            col.append(f"{name}_num")
            col.append(f"{name}_r")
            col.append(f"{name}_beta")
            col.append(f"{name}_alpha")
            col.append(f"{name}_nse")
    else:
        for name in df.columns[1:21]:
            col.append(f"{name}_kge")
            col.append(f"{name}_pbias")
            col.append(f"{name}_r2")
            col.append(f"{name}_num")
            col.append(f"{name}_r")
            col.append(f"{name}_beta")
            col.append(f"{name}_alpha")
            col.append(f"{name}_nse")
    metric_df = pd.DataFrame(columns=col)
    metric_df['staid'] = df_station['STAID']

    if not x_feature:
        x_feature = df.columns[21:]
    if exclude:
        x_feature = [x for x in x_feature if x not in exclude_features]

    # for staid in tqdm(train_station):
    #     metric_df = predict_single_station(root, staid, "train", metric_df, model_list, output_dir, temporal_transform, station_transform, target_detransform,
    #                            valid_threshold=valid_threshold, x_feature=x_feature, y_feature=y_feature, exclude=exclude, location_static=location_static)

    for staid in tqdm(test_station):
        metric_df = predict_single_station(root, staid, "test", metric_df, model_list, output_dir, temporal_transform, station_transform, target_detransform,
                               valid_threshold=valid_threshold, x_feature=x_feature, y_feature=y_feature, exclude=exclude, location_static=location_static)

    print("Saving metrics to ", output_dir, "/metric.csv")
    metric_dir = os.path.join(output_dir, "metric")
    metric_df.to_csv(os.path.join(metric_dir, "metric.csv"), index=False, na_rep='')

def predict_single_station(root, staid, split, metric_df, model_list, output_dir, temporal_transform, station_transform, target_detransform,
                           valid_threshold=51, x_feature=None, y_feature=None, exclude=False, location_static=False):
    with open(os.path.join(root, "static_filtered.csv")) as f:
        df_station = pd.read_csv(f, dtype={'STAID': str})

    df = pd.read_csv(os.path.join(root, f"{staid}.csv"))

    if y_feature:
        output_df = df[y_feature].copy()
        col_names = y_feature
    else:
        output_df = df.iloc[:, :21].copy()
        col_names = df.columns[1:21]

    for name in col_names:
        output_df[f"{name}_pred"] = np.nan
    for name in col_names:
        output_df[f"{name}_type"] = 3 # 默认x不全
    
    temporal = df.loc[:, x_feature].values

    if y_feature:
        target = df.loc[:, y_feature].values
    else:
        target = df.iloc[:, 1:21].values
    staid = staid.lstrip('0')
    station = df_station.loc[df_station['STAID'] == staid]
    if exclude:
        drop_columns = [f for f in exclude if f in station.columns]
        if location_static:
            # if "LAT_GAGE,LNG_GAGE" column is in drop_columns, we should keep it
            if 'LAT_GAGE' in drop_columns:
                drop_columns.remove('LAT_GAGE')
            if 'LNG_GAGE' in drop_columns:
                drop_columns.remove('LNG_GAGE')
        station = station.drop(columns=drop_columns)
    
    if location_static:
        # preserve only the "LAT_GAGE,LNG_GAGE" column in station, the rest of station is concatenated with temporal
        station_rest = station.drop(columns=['STAID', 'LAT_GAGE', 'LNG_GAGE'])
        station_rest = station_rest.values
        station_rest = np.repeat(station_rest, temporal.shape[0], axis=0)

        station = station[['LAT_GAGE', 'LNG_GAGE']].values
        station = np.repeat(station, temporal.shape[0], axis=0)

        temporal_date = df[['datenum', 'sinT', 'cosT']].values
        station = np.concatenate([temporal_date, station], axis=1)

        temporal = np.concatenate([temporal, station_rest], axis=1)

        x_feature_no_date = [f for f in x_feature if f not in ['datenum', 'sinT', 'cosT']]
        temporal_rest = df[x_feature_no_date].values
        temporal = np.concatenate([temporal_rest, station_rest], axis=1)
    else:
        station = station.iloc[:, 1:].values
    station = torch.from_numpy(station).float()
    # import pdb; pdb.set_trace()

    for idx, model in enumerate(model_list):
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            temporal_tensor = torch.from_numpy(temporal).float()
            temporal_tensor = temporal_transform(temporal_tensor)
            temporal_tensor = temporal_tensor.cuda()
            
            station_tensor = station.repeat(temporal_tensor.shape[0], 1)
            # station_tensor = np.repeat(station, temporal_tensor.shape[0], axis=0)
            station_tensor = station_transform(station)
            station_tensor = station_tensor.cuda()
            
            output = model(temporal_tensor, station_tensor)
            output = target_detransform(output)
            output = output.detach().cpu().numpy()

            for i, name in enumerate(col_names):
                if i == idx:
                    output_df[f"{name}_pred"] = output[:, i]
                    
                    # 2：x齐全，可以预测出来且不是0和1
                    mask = df.iloc[:, 21:].notna().all(axis=1)
                    # mask &= df.iloc[:, 1:21].isna().all(axis=1)
                    output_df.loc[mask, f"{name}_type"] = 2

                    mask &= df.loc[:, name].notna()
                    # 0：x齐全，有效的用于train的y
                    if split == "train":
                        # train_mask &= df.iloc[:, int(i+1)].notna().any(axis=1)
                        # train_mask = df.loc[:, name].notna()
                        # train_mask &= mask
                        output_df.loc[mask, f"{name}_type"] = 0
                    elif split == "test":
                        # 1：x齐全，有效的用于test的y
                        # 4: x齐全，有效的用于test的y，但是y数量太少
                        # test_mask &= df.iloc[:, int(i+1)].notna().any(axis=1)
                        # test_mask = df.loc[:, name].notna()
                        # test_mask &= mask

                        if mask.sum() < valid_threshold:
                            output_df.loc[mask, f"{name}_type"] = 4
                        else:
                            output_df.loc[mask, f"{name}_type"] = 1
                    
                    # since the whole station is either train or test data, we calculate metric on this station
                    test_targets = df.loc[mask, name].values
                    test_mask = mask.to_numpy()
                    test_pred = output[mask, i]
                    if (~np.isnan(test_targets)).sum() < valid_threshold:
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_num"] = (~np.isnan(test_targets)).sum()
                        continue
                    test_targets = torch.from_numpy(test_targets).float()
                    test_pred = torch.from_numpy(test_pred).float()
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_kge"] = _kge(test_pred, test_targets)
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_pbias"] = _pbias(test_pred, test_targets)
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_r2"] = _r_squared(test_pred, test_targets)
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_num"] = test_mask.sum()
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_r"] = _r(test_pred, test_targets)
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_beta"] = _beta(test_pred, test_targets)
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_alpha"] = _alpha(test_pred, test_targets)
                    metric_df.loc[metric_df['staid'] == staid, f"{name}_nse"] = _nse(test_pred, test_targets)
    output_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    with open(os.path.join(output_dir, f"{staid}.csv"), "w") as f:
        output_df.to_csv(f, index=False, na_rep='')
    return metric_df


if __name__ == "__main__":
    path = "saved_deepOnetC_wo_tmmx_etr/models/Mnist_LeNet/0717_090407/best_target_state_dict.pth"
    config_path = "saved_deepOnetC_wo_tmmx_etr/models/Mnist_LeNet/0717_090407/config.json"
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('-c', '--config', default=config_path, type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    config = ConfigParser.from_args(args)

    data_class = getattr(dataset, config['dataset'])
    train_data = data_class("../climate_washed", config['normalize'], split="train", x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'])
    stats = train_data.get_stats()

    activation = getattr(nn, config["activation"])
    trunk = config.init_obj('trunk', module_model, activation=activation)
    branch = config.init_obj('branch', module_model, activation=activation)
    z_net = config.init_obj('z_net', module_model, activation=activation)

    if config['arch_name'] == 'TriDeepONet':
        model = module_model.TriDeepONet(trunk, branch, z_net)
    elif config['arch_name'] == 'DeepONet':
        model = module_model.DeepONet(trunk, branch)
    
    state_dicts = torch.load(path)

    model_states = []
    for s in range(len(state_dicts)):
        model_states.append(state_dicts[s])
    target_stat = stats['target']
    target_detransform = LogDeNormalize(
        target_stat
    )
    
    kge_list = []
    r2_list = []
    pbias_list = []
    
    val_data = data_class("../climate_washed", config['normalize'], split="test", stats=stats, x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'])
    # for i in tqdm(range(len(val_data))):
    #     trunk, branch, target = val_data[i]
    #     trunk, branch, target = trunk.cuda(), branch.cuda(), target.cuda()
    #     final_prediction = torch.full(target.shape, torch.nan, device=target.device)
        
        # with open("saved_deepOnetA/models/Mnist_LeNet/0529_173020/model_best.pth", 'rb') as f:
        #     state = torch.load(f)
        #     model.load_state_dict(state)
        # model = model.cuda()
        # model.eval()
        # with torch.no_grad():
        #     output = model(trunk, branch)
        #     final_prediction = output
        
    #     for idx, state in enumerate(model_states):
    #         model.load_state_dict(state)
    #         model.eval()
    #         model = model.cuda()
            
    #         with torch.no_grad():
    #             output = model(trunk, branch)
    #             final_prediction[:, idx] = output[:, idx]
    #     final_prediction = target_detransform(final_prediction)
    #     target = target_detransform(target)
    #     valid_count = (~torch.isnan(target)).sum(axis=0).cpu().numpy()
        
    #     kge_temp = kge(final_prediction, target).numpy()
    #     r2_temp = r_squared(final_prediction, target).numpy()
    #     pbias_temp = pbias(final_prediction, target).numpy()
        
    #     kge_temp = np.where(valid_count < 2, np.nan, kge_temp)
    #     r2_temp = np.where(valid_count < 2, np.nan, r2_temp)
    #     pbias_temp = np.where(valid_count < 2, np.nan, pbias_temp)
        
    #     kge_list.append(kge_temp)
    #     r2_list.append(r2_temp)
    #     pbias_list.append(pbias_temp)

    # final_kge = np.array(kge_list)
    # final_r2 = np.array(r2_list)
    # final_pbias = np.array(pbias_list)
    # print("Final KGE: ", np.nanmedian(final_kge, axis=0))
    # print("Final R2: ", np.nanmedian(final_r2, axis=0))
    # print("Final Pbias: ", np.nanmedian(final_pbias, axis=0))

    # with open("saved_deepOnetA/models/Mnist_LeNet/0529_173020/model_best.pth", 'rb') as f:
    #     state = torch.load(f)
    dir = os.path.dirname(path)
    output_dir = os.path.join(dir, "predictions")
    predict_whole_dataset(model, model_states, stats, valid_threshold=config['metric_threshold'], 
                          normalize=config['normalize'], split='split_datesC.txt', output_dir=output_dir,
                          x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'])