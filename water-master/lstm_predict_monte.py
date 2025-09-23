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
from model.model import LSTM
from parse_config import ConfigParser
import argparse
import json

def run_monte_carlo_predictions(model, state_dicts, stats, config, n_samples=30, **kwargs):
    """
    Runs Monte Carlo predictions by calling predict_whole_dataset multiple times
    
    Args:
        model: LSTM model
        state_dicts: model state dictionaries
        stats: dataset statistics
        config: configuration dictionary
        n_samples: number of Monte Carlo samples
        **kwargs: additional arguments for predict_whole_dataset
    """
    base_output_dir = kwargs.get('output_dir', 'output')
    monte_carlo_dir = os.path.join(base_output_dir, 'monte_carlo')
    if not os.path.exists(monte_carlo_dir):
        os.makedirs(monte_carlo_dir)
    
    print(f"Running {n_samples} Monte Carlo predictions...")
    print(f"Results will be saved to {monte_carlo_dir}")

    # Create list of models and set to train mode for dropout
    model_list = []
    for i, state in enumerate(state_dicts):
        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict(state)
        model_copy.train()  # Enable dropout
        model_list.append(model_copy)
    
    for sample in tqdm(range(n_samples)):
        print(f"Running Monte Carlo sample {sample}...")
        # Create sample-specific output directory
        sample_dir = os.path.join(monte_carlo_dir, f'sample_{sample}')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Update kwargs with new output directory
        kwargs['output_dir'] = sample_dir
        
        # Run prediction
        predict_whole_dataset(model_list, stats, config, **kwargs)
        torch.cuda.empty_cache()
    # Aggregate results
    print("Aggregating Monte Carlo results...")
    aggregate_monte_carlo_results(monte_carlo_dir, n_samples, config['y_feature'])


def aggregate_monte_carlo_results(monte_carlo_dir, n_samples, y_feature):
    """
    Aggregates results from multiple Monte Carlo runs
    
    Args:
        monte_carlo_dir: directory containing Monte Carlo results
        n_samples: number of Monte Carlo samples
        y_feature: list of target features
    """
    # Initialize storage for all metrics
    all_metrics = []
    
    # Read metrics from each sample
    for sample in range(n_samples):
        sample_dir = os.path.join(monte_carlo_dir, f'sample_{sample}', 'metric', 'metrics.csv')
        if os.path.exists(sample_dir):
            df = pd.read_csv(sample_dir)
            all_metrics.append(df)
    
    # Stack all dataframes
    stacked_metrics = pd.concat(all_metrics, axis=0, keys=range(n_samples))
    
    # Calculate statistics across samples
    stats_df = pd.DataFrame()
    stats_df['staid'] = all_metrics[0]['staid']
    
    # Calculate mean and std for each metric
    for feat in y_feature:
        for metric in ['kge', 'pbias', 'r2']:
            col = f"{feat}_{metric}"
            
            # Mean
            mean_values = stacked_metrics.groupby(level=1)[col].mean()
            stats_df[f"{col}_mean"] = mean_values.values
            
            # Standard deviation
            std_values = stacked_metrics.groupby(level=1)[col].std()
            stats_df[f"{col}_std"] = std_values.values
            
            # 95% confidence interval
            ci_values = 1.96 * std_values
            stats_df[f"{col}_95ci"] = ci_values.values
    
    # Save aggregated results
    stats_df.to_csv(os.path.join(monte_carlo_dir, 'aggregated_metrics.csv'), index=False)
    print(f"Aggregated results saved to {os.path.join(monte_carlo_dir, 'aggregated_metrics.csv')}")


def predict_whole_dataset(model_list, stats, config, valid_threshold=51, normalize='MeanStdNormalize',
                          split='split_datesA.txt', root="../climate_new", output_dir="output",
                          x_feature=None, y_feature=None, exclude=0, n_samples=10):
    """0：有效的用于train的y 1：有效的用于test的y 2：x齐全可以预测出来且不是0和1 3：x不全 4:有效的用于test的y,但是y数量太少"""
    print("Calculating metrics with Monte Carlo Dropout...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metric_dir = os.path.join(output_dir, "metric")
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    with open("../group.json") as f:
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
    #     model_copy.train()
    #     model_list.append(copy.deepcopy(model_copy))
    
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
            # col.append(f"{name}_std")
            # col.append(f"{name}_95ci")
            # col.append(f"{name}_num")
    else:
        for name in df.columns[1:21]:
            col.append(f"{name}_kge")
            col.append(f"{name}_r2")
            col.append(f"{name}_pbias")
            # col.append(f"{name}_std")
            # col.append(f"{name}_95ci")
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
    uncertainties = defaultdict(dict)
    for staids, date, x, y in tqdm(loader):
        x, y = x.cuda(), y.cuda()
        # import pdb; pdb.set_trace()
        
        prediction = torch.zeros((y.shape[0], y.shape[2]))
        for idx, model in enumerate(model_list):
            # model.eval()
            model = model.cuda()
            with torch.no_grad():
                output = model(x)
                output = output[:, -1, :]
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

        with open(f"../climate_washed/{staid}.csv") as f:
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
            metric_df.loc[metric_df['staid'] == staid, f"{name}_kge"] = full_kge[i]
            metric_df.loc[metric_df['staid'] == staid, f"{name}_pbias"] = full_pbias[i]
            metric_df.loc[metric_df['staid'] == staid, f"{name}_r2"] = full_r2[i]
            metric_df.loc[metric_df['staid'] == staid, f"{name}_num"] = valid_count[i]

    metric_df.to_csv(os.path.join(metric_dir, "metrics.csv"), index=False, na_rep='')
    return

if __name__ == "__main__":
    path = "/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax_y/baseline_dropout_0.3/models/lstm/1016_145644/best_target_state_dict.pth"
    config = "/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax_y/baseline_dropout_0.3/models/lstm/1016_145644/config.json"
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('-c', '--config', default=config, type=str,
                      help='config file path (default: config_LSTM.json)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    config = ConfigParser.from_args(args)
    
    data_class = getattr(dataset, config['dataset'])
    train_data = data_class("../climate_new", config['normalize'], split="train",
                            x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'], minmax_feature=config['minmax_feature'])
    stats = train_data.get_stats()
    
    model = LSTM(**config['arch'])
    state_dicts = torch.load(path)

    model_states = []
    for s in range(len(state_dicts)):
        model_states.append(state_dicts[s])

    dir = os.path.dirname(path)
    output_dir = os.path.join(dir, "predictions")

    split_dict = {
        "ClimateDatasetV2A": "split_datesA.txt",
        "ClimateDatasetV2B": "split_datesB.txt",
        "ClimateDatasetV2C": "split_datesC.txt",
    }

    # predict_whole_dataset(model, model_states, stats, config, config['metric_threshold'], config['normalize'], split="split_datesC.txt",
    #                       output_dir=output_dir, x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'])
    run_monte_carlo_predictions(
        model=model,
        state_dicts=model_states,
        stats=stats,
        config=config,
        n_samples=50,  # Number of Monte Carlo samples
        valid_threshold=config['metric_threshold'],
        normalize=config['normalize'],
        split=split_dict[config['dataset']],
        output_dir=output_dir,
        x_feature=config['x_feature'],
        y_feature=config['y_feature'],
        exclude=config['exclude'],
        root="../climate_new",
    )