import os
import pandas as pd
from mytransform import MeanStdNormalize, MinMaxNormalize, MeanStdDeNormalize, MinMaxDeNormalize, LogDeNormalize
from model.metric import _nse, _pbias, _r_squared, _kge
from model.metric import pbias, r_squared, kge
import torch
import numpy as np
import copy
from tqdm import tqdm
import lstm_dataset as dataset
from torch.utils.data import DataLoader
from collections import defaultdict
from model.model import LSTM, InformerWrapper
from parse_config import ConfigParser
import argparse
import json

def _prepare_informer_inputs(x, label_len=96):
    """
    Prepare encoder and decoder inputs for Informer prediction
    Simplified version: treat x as a whole, use dummy time features
    
    Args:
        x: input data [batch_size, time_len, features]
        label_len: length of label sequence for decoder (default 96)
    Returns:
        x_enc, x_mark_enc, x_dec, x_mark_dec
    """
    batch_size, time_len = x.shape[0], x.shape[1]
    
    # Use x as a whole
    x_data = x  # [B, time_len, features]
    
    # Encoder inputs: use all timesteps EXCEPT the last one
    x_enc = x_data[:, :-1, :]  # [B, time_len-1, features]
    
    # Create dummy time features (zeros) for encoder
    time_features = 3  # Standard for daily frequency
    x_mark_enc = torch.zeros(batch_size, time_len-1, time_features).to(x.device)
    
    # Decoder inputs: for prediction, we need label_len + pred_len
    label_len = min(label_len, time_len - 1)  # Don't exceed available data
    pred_len = 1
    
    # Label part: last label_len timesteps from encoder sequence
    start_idx = max(0, time_len - 1 - label_len)
    x_dec_label = x_data[:, start_idx:time_len-1, :]  # [B, label_len, features]
    
    # Prediction part: use actual input data for the timestep we're predicting
    x_dec_pred = x_data[:, -1:, :]  # [B, 1, features] - actual data for prediction timestep
    
    # Concatenate label and prediction parts
    x_dec = torch.cat([x_dec_label, x_dec_pred], dim=1)  # [B, label_len + 1, features]
    
    # Create dummy time features (zeros) for decoder
    x_mark_dec = torch.zeros(batch_size, x_dec.shape[1], time_features).to(x.device)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec

def _is_informer_model(model):
    """Check if the model is an Informer model"""
    return isinstance(model, InformerWrapper)

def predict_whole_dataset(model, state_dicts, stats, config, valid_threshold=51, normalize='MeanStdNormalize',
                          split='split_datesA.txt', root="../climate_new", output_dir="output",
                          x_feature=None, y_feature=None, exclude=0):
    """0：有效的用于train的y 1：有效的用于test的y 2：x齐全可以预测出来且不是0和1 3：x不全 4:有效的用于test的y,但是y数量太少"""
    print("Calculating metrics...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metric_dir = os.path.join(output_dir, "metric")
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    with open("group.json") as f:
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
    df_station['STAID'] = df_station['STAID'].apply(lambda x: x.zfill(8))
    
    assert y_feature is not None, "y_feature must be specified"
    target_minmax_idx = [i for i, f in enumerate(y_feature) if f in config['minmax_feature']]
    target_stat = stats['target']
    target_detransform = LogDeNormalize(
        target_stat,
        target_minmax_idx,
    )
    
    col = ['staid']
    df = pd.read_csv(os.path.join(root, f"01054200.csv"))
    if y_feature:
        for name in y_feature:
            col.append(f"{name}_kge")
            col.append(f"{name}_pbias")
            col.append(f"{name}_r2")
            # col.append(f"{name}_num")
    else:
        for name in df.columns[1:21]:
            col.append(f"{name}_kge")
            col.append(f"{name}_r2")
            col.append(f"{name}_pbias")
            # col.append(f"{name}_num")
    col_names = y_feature if y_feature else df.columns[1:21]
    metric_df = pd.DataFrame(columns=col)
    metric_df['staid'] = df_station['STAID']
    
    valset = getattr(dataset, config['dataset'])
    root_dir = config['root_dir'] if config['root_dir'] is not None else "../climate_new"
    data = valset(root_dir, config['normalize'], split="test", stats=stats, seqlen=365,
                  x_feature=x_feature, y_feature=y_feature, exclude=exclude, retDate=True, minmax_feature=config['minmax_feature'])
    # data = valset("../climate_new", config['normalize'], split="test", stats=stats, testNum=20)
    loader = DataLoader(data, batch_size=1024, num_workers=12, pin_memory=True)
    
    if x_feature:
        if exclude:
            x_feature = [f for f in x_feature if f not in exclude_features]
        x_feat = len(x_feature)
    else:
        x_feat = 26
    
    targets = defaultdict(dict)
    predictions = defaultdict(dict)
    masks = defaultdict(dict)
    for staids, date, x, y in tqdm(loader):
        x, y = x.cuda(), y.cuda()
        # import pdb; pdb.set_trace()
        
        prediction = torch.zeros((y.shape[0], y.shape[2]))
        for idx, model in enumerate(model_list):
            model.eval()
            model = model.cuda()
            with torch.no_grad():
                if _is_informer_model(model):
                    # Informer model: prepare encoder/decoder inputs
                    label_len = config.get('label_len', 96)  # Get from config or use default
                    x_enc, x_mark_enc, x_dec, x_mark_dec = _prepare_informer_inputs(x, label_len)
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    output = output[:, -1, :]  # [B, pred_len, c_out] -> [B, c_out] (last prediction timestep)
                else:
                    # LSTM model: standard forward pass
                    output = model(x)
                    output = output[:, -1, :]  # [B, seq_len, out_dim] -> [B, out_dim]
                
                output = output.detach().cpu()
                prediction[:, idx] = output[:, idx]

        target = y[:, -1, :].cpu()
        x = x[:, -1, :x_feat].cpu()

        mask = ~(x == -1).any(dim=1)
        
        for idx, staid in enumerate(staids):
            staid = staid.zfill(8)
            # targets[staid].append(target[idx])
            # predictions[staid].append(prediction[idx])
            # masks[staid].append(mask[idx])
            targets[staid][date[idx]] = target[idx]
            predictions[staid][date[idx]] = prediction[idx]
            masks[staid][date[idx]] = mask[idx]

    # import pickle
    # with open("results.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             'targets': targets,
    #             'predictions': predictions,
    #             'masks': masks
    #         },
    #         f
    #     )

    # enumerate through each station
    for staid in targets.keys():
        # import pdb; pdb.set_trace()
        staid = staid.zfill(8)
        # target = torch.stack(targets[staid], axis=0).cuda()
        # pred = torch.stack(predictions[staid], axis=0).cuda()
        # mask = torch.stack(masks[staid], axis=0).cuda()
        target = torch.stack(list(targets[staid].values()), axis=0)
        pred = torch.stack(list(predictions[staid].values()), axis=0)
        mask = torch.stack(list(masks[staid].values()), axis=0)
        dates = list(targets[staid].keys())

        with open(f"climate_washed/{staid}.csv") as f:
            washed_df = pd.read_csv(f)
        allowed_dates = washed_df['Date'].tolist()
        # find the intersection between dates and allowed_dates
        mask2 = torch.tensor([d in allowed_dates for d in dates])
        mask &= mask2

        target = target[mask]
        pred = pred[mask]

        # print((target != -1).sum(axis=0))
        # target = torch.where(target == -1, torch.tensor(float('nan')), target)
        
        pred = target_detransform(pred).cuda()
        target = target_detransform(target).cuda()

        full_kge = kge(pred, target)
        full_pbias = pbias(pred, target)
        full_r2 = r_squared(pred, target)

        valid_count = (~torch.isnan(target)).sum(axis=0).cpu()
        full_kge = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_kge)
        full_pbias = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_pbias)
        full_r2 = torch.where(valid_count < valid_threshold, torch.tensor(float('nan')), full_r2)

        for i, name in enumerate(col_names):
            metric_df.loc[metric_df['staid'] == staid, f"{name}_kge"] = full_kge[i].item()
            metric_df.loc[metric_df['staid'] == staid, f"{name}_pbias"] = full_pbias[i].item()
            metric_df.loc[metric_df['staid'] == staid, f"{name}_r2"] = full_r2[i].item()
            metric_df.loc[metric_df['staid'] == staid, f"{name}_num"] = valid_count[i].item()

    metric_df.to_csv(os.path.join(metric_dir, "metrics.csv"), index=False, na_rep='')
    return

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
    for name in df.columns[1:21]:
        col.append(f"{name}_kge")
        col.append(f"{name}_pbias")
        col.append(f"{name}_r2")
    metric_df = pd.DataFrame(columns=col)
    metric_df['staid'] = df_station['STAID']

    for station_index, staid in enumerate(tqdm(df_station['STAID'])):
        # print("Saving results for station: ", staid)
        if len(staid) < 8:
            staid_str = '0' * (8 - len(staid)) + staid
        else:
            staid_str = staid
        df = pd.read_csv(os.path.join(root, f"{staid_str}.csv"))

        output_df = df.iloc[:, :21].copy()
        col_names = df.columns[1:21]

        for name in col_names:
            output_df[f"{name}_pred"] = np.nan
        for name in col_names:
            output_df[f"{name}_type"] = 3 # 默认x不全

        temporal = df.iloc[:, 21:].values
        target = df.iloc[:, 1:21].values
        station = df_station.loc[df_station['STAID'] == staid]
        station = station.iloc[:, 1:].values
        station = torch.from_numpy(station).float()

        for idx, model in enumerate(model_list):
            model.eval()
            model = model.cuda()
            with torch.no_grad():
                temporal_tensor = torch.from_numpy(temporal).float()
                temporal_tensor = temporal_transform(temporal_tensor)
                temporal_tensor[torch.isnan(temporal_tensor)] = -1
                temporal_tensor = temporal_tensor.cuda()
                
                station_tensor = station.repeat(temporal_tensor.shape[0], 1)
                # station_tensor = np.repeat(station, temporal_tensor.shape[0], axis=0)
                station_tensor = station_transform(station)
                station_tensor = station_tensor.cuda()
                x = torch.cat([temporal_tensor, station_tensor], dim=1)
                
                output = model(x)
                output = target_detransform(output)
                output = output.detach().cpu().numpy()
                
                for i, name in enumerate(col_names):
                    if i == idx:
                        output_df[f"{name}_pred"] = output[:, i]
                        
                        # 2：x齐全，可以预测出来且不是0和1
                        mask = df.iloc[:, 21:].notna().all(axis=1)
                        # mask &= df.iloc[:, 1:21].isna().all(axis=1)
                        output_df.loc[mask, f"{name}_type"] = 2

                        # 0：x齐全，有效的用于train的y
                        train_mask = df['Unnamed: 0'].isin(train_dates)
                        # train_mask &= df.iloc[:, int(i+1)].notna().any(axis=1)
                        train_mask &= df.iloc[:, int(i+1)].notna()
                        train_mask &= df.iloc[:, 21:].notna().all(axis=1)
                        output_df.loc[train_mask, f"{name}_type"] = 0
                        
                        # 1：x齐全，有效的用于test的y
                        # 4: x齐全，有效的用于test的y，但是y数量太少
                        test_mask = df['Unnamed: 0'].isin(test_dates)
                        # test_mask &= df.iloc[:, int(i+1)].notna().any(axis=1)
                        test_mask &= df.iloc[:, int(i+1)].notna()
                        test_mask &= df.iloc[:, 21:].notna().all(axis=1)
                        # if '1054200' in staid:
                        #     import pdb;pdb.set_trace()
                        if test_mask.sum() < valid_threshold:
                            output_df.loc[test_mask, f"{name}_type"] = 4
                        else:
                            output_df.loc[test_mask, f"{name}_type"] = 1
                        
                        # calculate metrics on test dataset
                        test_targets = df.loc[test_mask, name].values
                        test_mask = test_mask.to_numpy()
                        test_pred = output[test_mask, i]
                        if (~np.isnan(test_targets)).sum() < valid_threshold:
                            continue
                        test_targets = torch.from_numpy(test_targets).float()
                        test_pred = torch.from_numpy(test_pred).float()
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_kge"] = _kge(test_pred, test_targets)
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_pbias"] = _pbias(test_pred, test_targets)
                        metric_df.loc[metric_df['staid'] == staid, f"{name}_r2"] = _r_squared(test_pred, test_targets)
        # if station_index == 10:
            # break
        output_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        with open(os.path.join(output_dir, f"{staid_str}.csv"), "w") as f:
            output_df.to_csv(f, index=False, na_rep='')
    print("Saving metrics to ", metric_dir, "/metric.csv")
    metric_df.to_csv(os.path.join(metric_dir, "metric.csv"), index=False, na_rep='')


if __name__ == "__main__":
    path = "unified_ablation_lstm_logminmax_y/baseline_dropout_0.3_665/models/lstm/0929_144214/best_target_state_dict.pth"
    config = "unified_ablation_lstm_logminmax_y/baseline_dropout_0.3_665/models/lstm/0929_142604/config.json"
    # path = "unified_ablation_informer/baseline_dropout_0.1_665/models/informer/0929_144935/checkpoint-epoch70.pth"
    # config = "unified_ablation_informer/baseline_dropout_0.1_665/models/informer/0929_144935/config.json"
    # path = "/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax_y/baseline_dropout_0.3/models/lstm/1016_145644/best_target_state_dict.pth"
    # config = "/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax_y/baseline_dropout_0.3/models/lstm/1016_145644/config.json"
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('-c', '--config', default=config, type=str,
                      help='config file path (default: config_LSTM.json)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    config = ConfigParser.from_args(args)
    
    data_class = getattr(dataset, config['dataset'])
    root_dir = config['root_dir'] if config['root_dir'] is not None else "../climate_new"
    train_data = data_class(root_dir, config['normalize'], split="train", x_feature=config['x_feature'], 
                            y_feature=config['y_feature'], exclude=config['exclude'], minmax_feature=config['minmax_feature'])
    stats = train_data.get_stats()
    
    # Determine model type based on config
    if config.get('model_type') == 'informer':
        model = InformerWrapper(**config['arch'])
    else:
        model = LSTM(**config['arch'])
    
    state_dicts = torch.load(path)

    model_states = []
    for s in range(len(state_dicts)):
        model_states.append(state_dicts[s])

    dir = os.path.dirname(path)
    output_dir = os.path.join(dir, "predictions")
    print(output_dir)
    predict_whole_dataset(model, model_states, stats, config, config['metric_threshold'], config['normalize'], split="split_datesC.txt", root=root_dir,
                          output_dir=output_dir, x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'])
