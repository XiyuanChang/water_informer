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
from model.informer_enc_decval import LSTM, InformerWrapper
from parse_config import ConfigParser
import argparse
import json

def predict_whole_dataset(model, state_dicts, stats, config, valid_threshold=51, normalize='MeanStdNormalize',
                          split='split_datesA.txt', root="../climate_new", output_dir="output",
                          x_feature=None, y_feature=None, exclude=0, device=None):
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

    # model_list = []
    # for state in state_dicts:
    #     model_copy = copy.deepcopy(model)
    #     model_copy.load_state_dict(state)
    #     model_list.append(copy.deepcopy(model_copy))
    
    model_single = copy.deepcopy(model)
    model_single.load_state_dict(state_dicts, strict=True)
    model_single = model_single.eval()
    model_single.to(device)
    
    with open("climate_new/static_filtered.csv") as f:
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
    df = pd.read_csv("climate_new/01054200.csv")
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
    loader = DataLoader(data, batch_size=512, num_workers=12, pin_memory=True)
    
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
        x, y = x.to(device), y.to(device)
        prediction = torch.zeros((y.shape[0], y.shape[2]))
        
        with torch.no_grad():
            output = model_single(x)
            output = output[:, -1, :]
            output = output.detach().cpu()
            
            prediction = output

        target = y[:, -1, :].cpu()
        x = x[:, -1, :x_feat].cpu()

        # mask = ~(x == -1).any(dim=1)
        mask = ~(x[:, 20:] == 0).any(dim=1)
        
        for idx, staid in enumerate(staids):
            staid = staid.zfill(8)
            
            targets[staid][date[idx]] = target[idx]
            predictions[staid][date[idx]] = prediction[idx]
            masks[staid][date[idx]] = mask[idx]

    
    # enumerate through each station
    for staid in targets.keys():
        staid = staid.zfill(8)
        
        dates = sorted(list(targets[staid].keys()))
        target = torch.stack([targets[staid][d] for d in dates], axis=0)
        pred = torch.stack([predictions[staid][d] for d in dates], axis=0)
        mask = torch.stack([masks[staid][d] for d in dates], axis=0)

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
        
        pred = target_detransform(pred).to(device)
        target = target_detransform(target).to(device)
        
        ########### print pred, target shape
        print('staid: ', staid)
        print('pred shape: ', pred.shape)
        print('target shape: ', target.shape)
        dates_np = np.array(dates, dtype=object)    # dtype=object 保留字符串
        dates_kept = dates_np[mask.cpu().numpy()].tolist()
        print('dates:', dates_kept)

        
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

    #metric_df.to_csv(os.path.join(metric_dir, "metrics.csv"), index=False, na_rep='')
    #metric_df.to_csv(os.path.join(metric_dir, "metrics_climate_washed.csv"), index=False, na_rep='')
    #metric_df.to_csv(os.path.join(metric_dir, "metrics_N_N.csv"), index=False, na_rep='')
    #metric_df.to_csv(os.path.join(metric_dir, "metrics_N_N_decvalue.csv"), index=False, na_rep='')
    metric_df.to_csv(os.path.join(metric_dir, "metrics_N_N_decno97_0.csv"), index=False, na_rep='')
    return

    
if __name__ == "__main__":
    # config = "Informer_enc_dec/baseline/models/informer/1125_033547/config.json"
    #path = "Informer_enc_dec/baseline/models/informer/1125_033547/last_state_dict.pth"
    # config = "Informer_enc_decorg_val/models/informer/1129_192741/config_washed.json" #metrics_climate_washed
    # path = "Informer_enc_decorg_val/models/informer/1129_192741/last_state_dict.pth"
    # path = "Informer_enc365_dec96/models/informer/1202_022941/last_state_dict.pth"
    # config = "Informer_enc365_dec96/models/informer/1202_022941/config_new.json"
    config = "Informer_enc_decorg_val/models/informer/1129_192741/config.json" #metrics_N_N_decvalue
    path = "Informer_enc_decorg_val/models/informer/1129_192741/last_state_dict.pth"
    
    
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

    # Select model type (LSTM by default; Informer if specified in config)
    model_type = config.get('model_type', 'lstm')
    if isinstance(model_type, str):
        model_type = model_type.lower()

    if model_type == 'informer':
        seq_len = config.get('seq_len', 365)
        label_len = config.get('label_len', 96)
        pred_len = config.get('pred_len', 1)

        model = InformerWrapper(
            in_dim=config['arch']['in_dim'],
            out_dim=config['arch']['out_dim'],
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 4),
            e_layers=config.get('e_layers', 2),
            d_layers=config.get('d_layers', 1),
            d_ff=config.get('d_ff', 2048),
            dropout=config['arch'].get('dropout', 0.1),
            attn=config.get('attn', 'prob'),
            embed=config.get('embed', 'timeF'),
            freq=config.get('freq', 'h'),
            activation=config.get('activation', 'gelu'),
            factor=config.get('factor', 5),
            distil=config.get('distil', True),
            mix=config.get('mix', True),
            output_attention=config.get('output_attention', False),
        )
    else:
        model = LSTM(**config['arch'])

    print(config['arch'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('num GPUs:', torch.cuda.device_count())
        print('using device:', torch.cuda.get_device_name(0))
    state_dict = torch.load(config.get('path', path), map_location='cpu')

    # model_states = []
    # for s in range(len(state_dicts)):
    #     model_states.append(state_dicts[s])
    
    
    dir = os.path.dirname(path)
    output_dir = os.path.join(dir, "predictions")
    print(output_dir)
    predict_whole_dataset(model, state_dict, stats, config, config['metric_threshold'], config['normalize'], split="split_datesC.txt", root=root_dir,
                          output_dir=output_dir, x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'],device=device)
