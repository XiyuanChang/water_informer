import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor
import warnings
import random
import copy
import torch
import argparse
import torch.nn.functional as F
import sys
import torchvision
import torch.nn as nn
sys.path.append('.//MyProject')

from model.model import LSTM
from parse_config import ConfigParser
import lstm_dataset as dataset
warnings.filterwarnings('ignore')

def get_training_dates(root_dir, split_file='split_datesC.txt', split='train'):
    """Get training dates from split file."""
    with open(os.path.join(root_dir, split_file), "r") as f:
        return [line.split()[0].strip() for line in f if line.split()[1].strip() == split]

def get_feature_bounds(root_dir, station_ids, training_dates, feature_cols):
    """Calculate global min/max for each feature across all stations using only training data."""
    feature_mins = {}
    feature_maxs = {}

    for station_id in tqdm(station_ids, desc="Calculating feature bounds"):
        file_path = os.path.join(root_dir, f"{station_id}.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        # Filter for training dates only
        train_mask = df.iloc[:, 0].isin(training_dates)
        df_train = df[train_mask]

        for col in feature_cols:
            valid_values = df_train[col].dropna()
            if len(valid_values) == 0:
                continue

            curr_min = valid_values.min()
            curr_max = valid_values.max()

            if col not in feature_mins or curr_min < feature_mins[col]:
                feature_mins[col] = curr_min
            if col not in feature_maxs or curr_max > feature_maxs[col]:
                feature_maxs[col] = curr_max

    return feature_mins, feature_maxs

def load_static_feature(station_id, static_df):
    # Load static features for the station and return as a numpy array
    static_feature = static_df[static_df['STAID'] == station_id]
    static_feature = static_feature.iloc[:, 1:].values
    return static_feature

def preprocess_x(df: pd.DataFrame, transforms: torchvision.transforms.Compose = None):
    # Preprocess x features
    data_tensor = torch.tensor(df.values, dtype=torch.float32)

    # Apply transforms if provided
    if transforms is not None:
        data_tensor = transforms(data_tensor)
    else:
        data_tensor = x_transform(data_tensor)

    # Convert tensor back to DataFrame
    df_transformed = pd.DataFrame(data_tensor.numpy(), columns=df.columns)
    
    df_transformed = df_transformed.fillna(-1)
    return df_transformed


def preprocess_y(df: pd.DataFrame, transforms: torchvision.transforms.Compose = None):
    # Preprocess y features
    data_tensor = torch.tensor(df.values, dtype=torch.float32)

    # Apply transforms if provided
    if transforms is not None:
        data_tensor = transforms(data_tensor)
    else:
        data_tensor = y_transform(data_tensor)

    # Convert tensor back to DataFrame
    df_transformed = pd.DataFrame(data_tensor.numpy(), columns=df.columns)

    return df_transformed

def unpreprocess_x(data):
    batched_detransform = torch.vmap(x_detransform)
    return batched_detransform(data)

def process_station(args):
    """Process a single station file. Designed for parallel processing."""
    (station_id, root_dir, output_dir, training_dates, feature_mins,
        feature_maxs, attack_config, _) = args
    input_path = os.path.join(root_dir, f"{station_id}.csv")
    output_path = os.path.join(output_dir, f"{station_id}.csv")

    static_df = pd.read_csv(os.path.join(root_dir, 'static_filtered.csv'), dtype={'STAID': str})
    static_df['STAID'] = static_df['STAID'].apply(lambda x: str(x).zfill(8))

    if not os.path.exists(input_path):
        return
    # Read the data
    df = pd.read_csv(input_path)
    df.columns.values[0] = ''
    train_mask = df.iloc[:, 0].isin(training_dates)
    df_train = df[train_mask].copy()
    original_indices = df_train.index  # Keep original indices
    if df_train.empty:
        print("No training data found for station", station_id)
        df.to_csv(output_path, index=False, na_rep="")
        return
    # Preprocess x and y
    x = preprocess_x(df_train[attack_config['x_feature_cols']])
    y = preprocess_y(df_train[attack_config['y_feature_cols']])
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # Find indices with at least one y subfeature with no nan values
    valid_y_mask = y.notnull().any(axis=1)
    valid_indices = valid_y_mask[valid_y_mask].index.tolist()
    # randomly sample 10% of valid indices
    valid_indices = random.sample(valid_indices, int(len(valid_indices) * 0.2))
    if len(valid_indices) == 0:
        print("Not enough valid indices, skipping station", station_id)
        df.to_csv(output_path, index=False, na_rep="")
        return
    # Load static features
    static_feature = load_static_feature(station_id, static_df)
    if static_feature is None:
        print("No static features found for station", station_id)
        df.to_csv(output_path, index=False, na_rep="")
        return
    # Set attack parameters
    epsilon = attack_config['epsilon']
    alpha = attack_config['alpha']
    num_iterations = attack_config['num_iterations']
    # Prepare bounds tensors for clamping
    x_feature_mins = torch.tensor([feature_mins[feat] for feat in attack_config['x_feature_cols']], dtype=torch.float32).view(1, 1, -1)
    x_feature_maxs = torch.tensor([feature_maxs[feat] for feat in attack_config['x_feature_cols']], dtype=torch.float32).view(1, 1, -1)
    normalized_x_feature_mins = x_transform(x_feature_mins[0]).view(1, 1, -1)
    normalized_x_feature_maxs = x_transform(x_feature_maxs[0]).view(1, 1, -1)

    x_feature_mins = x_feature_mins.cuda()
    x_feature_maxs = x_feature_maxs.cuda()
    normalized_x_feature_mins = normalized_x_feature_mins.cuda()
    normalized_x_feature_maxs = normalized_x_feature_maxs.cuda()
    # Prepare bounds for y features if attack_y is True
    if attack_config['attack_y']:
        y_feature_mins = torch.tensor([feature_mins[feat] for feat in attack_config['y_feature_cols']], dtype=torch.float32).view(1, -1)
        y_feature_maxs = torch.tensor([feature_maxs[feat] for feat in attack_config['y_feature_cols']], dtype=torch.float32).view(1, -1)
        normalized_y_feature_mins = y_transform(y_feature_mins)
        normalized_y_feature_maxs = y_transform(y_feature_maxs)
        y_feature_mins = y_feature_mins.cuda()
        y_feature_maxs = y_feature_maxs.cuda()
        normalized_y_feature_maxs = normalized_y_feature_maxs.cuda()
        normalized_y_feature_mins = normalized_y_feature_mins.cuda()
    # Prepare sequences and targets
    sequences = []
    targets = []
    sequence_indices = []
    for idx in valid_indices:
        idx = int(idx)  # Ensure idx is an integer
        if idx < 364:
            pad_length = 364 - idx
            pad_x = np.full((pad_length, len(attack_config['x_feature_cols'])), -1)
            seq_x = x.iloc[:idx+1].to_numpy()
            seq_x = np.concatenate([pad_x, seq_x], axis=0)
        else:
            seq_x = x.iloc[idx-364:idx+1].to_numpy()
        seq_y_out = y.iloc[idx].to_numpy()
        static_seq = np.tile(static_feature, (365, 1))
        input_seq = np.concatenate([seq_x, static_seq], axis=1)
        sequences.append(input_seq)
        targets.append(seq_y_out)
        sequence_indices.append(idx)
    if len(sequences) == 0:
        print("No valid sequences found for station", station_id)
        df.to_csv(output_path, index=False, na_rep="")
        return
    sequences = torch.tensor(np.array(sequences), dtype=torch.float32).cuda()
    targets = torch.tensor(np.array(targets), dtype=torch.float32).cuda()
    # Create adversarial versions of sequences and targets
    sequences_adv = sequences.clone().detach().requires_grad_(True)
    if sequences_adv.grad is None:
        sequences_adv.grad = torch.zeros_like(sequences_adv)
    else:
        sequences_adv.grad.zero_()
    if attack_config['attack_y']:
        targets_adv = targets.clone().detach().requires_grad_(True)
    else:
        targets_adv = targets.clone().detach()

    for iteration in range(num_iterations):
        model.zero_grad()
        if sequences_adv.grad is not None:
            sequences_adv.grad.detach_()
            sequences_adv.grad.zero_()
        outputs = model(sequences_adv)
        # valid_targets_mask should be 1 if the target is not nan
        valid_targets_mask = ~torch.isnan(targets_adv)
        outputs = outputs[:, -1, :]
        outputs_valid = outputs[valid_targets_mask]
        targets_adv_valid = targets_adv[valid_targets_mask]
        loss = ((outputs_valid - targets_adv_valid) ** 2).mean()

        if valid_targets_mask.sum() == 0:
            break  # Avoid division by zero
        
        loss.backward()
        # import pdb; pdb.set_trace()
        grad_sequences = sequences_adv.grad
        if torch.isnan(grad_sequences).any():
            print("Gradients contain NaN. Skipping this iteration.")
            continue # revised
        with torch.no_grad():
            x_start = 0
            x_end = len(attack_config['x_feature_cols'])
            # Update sequences (x) if attack_x is True
            if attack_config['attack_x']:
                grad_x = grad_sequences[:, :, x_start:x_end]
                # Handle attack on specific features
                if 'attack_feature_indices_x' in attack_config and attack_config['attack_feature_indices_x'] is not None:
                    attack_indices_x = attack_config['attack_feature_indices_x']
                    grad_mask = torch.zeros_like(grad_x)
                    grad_mask[:, :, attack_indices_x] = 1
                    grad_x = grad_x * grad_mask
                    sequences_adv[:, :, x_start:x_end] += alpha * grad_x.sign()
                else:
                    sequences_adv[:, :, x_start:x_end] += alpha * grad_x.sign()

                # Apply perturbation constraints
                delta_sequences = torch.clamp(sequences_adv - sequences, min=-epsilon, max=epsilon)
                sequences_adv[:, :, x_start:x_end] = sequences[:, :, x_start:x_end] + delta_sequences[:, :, x_start:x_end]

                # Clamp to original feature range
                sequences_adv[:, :, x_start:x_end] = torch.max(
                    torch.min(sequences_adv[:, :, x_start:x_end], normalized_x_feature_maxs),
                    normalized_x_feature_mins
                )
            else:
                sequences_adv[:, :, x_start:x_end] = sequences[:, :, x_start:x_end]

            # Update targets (y) if attack_y is True
            if attack_config['attack_y']:
                grad_targets = targets_adv.grad.data
                # Handle attack on specific y features
                if ('attack_feature_indices_y' in attack_config and
                        attack_config['attack_feature_indices_y'] is not None):
                    attack_indices_y = attack_config['attack_feature_indices_y']
                    grad_mask_y = torch.zeros_like(grad_targets)
                    grad_mask_y[:, attack_indices_y] = 1
                    grad_targets = grad_targets * grad_mask_y
                targets_adv = targets_adv + alpha * grad_targets.sign()
                delta_targets = torch.clamp(targets_adv - targets, min=-epsilon, max=epsilon)
                targets_adv = targets + delta_targets
                # Clamp to original feature range
                targets_adv = torch.max(
                    torch.min(targets_adv, normalized_y_feature_maxs),
                    normalized_y_feature_mins
                )
                if targets_adv.grad is None:
                    targets_adv.grad = torch.zeros_like(targets_adv)
                else:
                    targets_adv.grad.zero_()
    sequences_adv = sequences_adv.detach().cpu()
    if attack_config['attack_y']:
        targets_adv = targets_adv.detach().cpu()
    else:
        targets_adv = targets.detach().cpu()
    
    # denormalize x and y
    if attack_config['attack_x']:
        sequences_adv = sequences_adv[:, :, x_start:x_end]
        sequences_adv = unpreprocess_x(sequences_adv)
    sequences_adv = sequences_adv.numpy()
    if attack_config['attack_y']:
        targets_adv = y_detransform(targets_adv)
    targets_adv = targets_adv.numpy()

    # Write adversarial examples back to DataFrame
    for i, idx in enumerate(sequence_indices):
        idx = int(idx)
        adv_seq = sequences_adv[i]
        adv_target = targets_adv[i]
        if idx < 364:
            pad_length = 364 - idx
            df_positions = range(0, idx+1)
            adv_seq = adv_seq[pad_length:]  # Remove padding
        else:
            df_positions = range(idx-364, idx+1)
        df_positions = list(df_positions)
        if attack_config['attack_x']:
            adv_x_seq = adv_seq[:, x_start:x_end]
            # Map positions to original indices in df_train
            df_indices_in_df_train = [pos for pos in df_positions if pos < len(df_train)]
            if len(df_indices_in_df_train) > 0:
                original_df_indices = original_indices[df_indices_in_df_train]
                # Update the original DataFrame
                df.loc[original_df_indices, attack_config['x_feature_cols']] = adv_x_seq[-len(df_indices_in_df_train):]
        if attack_config['attack_y']:
            # Update the target values in the DataFrame
            df_idx = original_indices[idx]
            df.loc[df_idx, attack_config['y_feature_cols']] = adv_target
    df.to_csv(output_path, index=False, na_rep="")

def main(root_dir="climate_new", output_dir="climate_new_adv", epsilon=0.1, alpha=0.01,
         num_iterations=10, attack_x=True, attack_y=False, 
         attack_x_features=None, attack_y_features=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Copy non-CSV files
    for file in os.listdir(root_dir):
        if not file.endswith('.csv'):
            shutil.copy2(
                os.path.join(root_dir, file),
                os.path.join(output_dir, file)
            )
        elif 'static_filtered' in file:
            shutil.copy2(
                os.path.join(root_dir, file),
                os.path.join(output_dir, file)
            )
    # Get training dates
    training_dates = get_training_dates(root_dir)
    # Get station IDs
    static_df = pd.read_csv(os.path.join(root_dir, 'static_filtered.csv'))
    static_df['STAID'] = static_df['STAID'].apply(lambda x: str(x).zfill(8))

    station_ids = static_df['STAID'].astype(str).tolist()
    station_ids = [id.zfill(8) for id in station_ids]
    # Define feature columns
    x_feature_cols = ["runoff", "pr", "sph", "srad", "tmmn", "tmmx", "pet", "etr", "ph", "Conduc", 
                    "Ca", "Mg", "K", "Na", "NH4", "NO3", "Cl", "SO4", "distNTN", "LAI", "FAPAR", 
                    "NPP","datenum","sinT","cosT"]
    y_feature_cols = ["00010", "00095", "00300", "00400", "00405", "00600", "00605", "00618", 
                     "00660", "00665", "00681", "00915", "00925", "00930", "00935", "00940", 
                     "00945", "00955", "71846", "80154"]

    attack_feature_indices_x = [x_feature_cols.index(i) for i in attack_x_features] if attack_x_features else None
    attack_feature_indices_y = [y_feature_cols.index(i) for i in attack_y_features] if attack_y_features else None
    
    # Configure attack parameters
    attack_config = {
        'epsilon': epsilon,
        'alpha': alpha,
        'num_iterations': num_iterations,
        'attack_x': attack_x,
        'attack_y': attack_y,
        'x_feature_cols': x_feature_cols,
        'y_feature_cols': y_feature_cols,
        'attack_feature_indices_x': attack_feature_indices_x,
        'attack_feature_indices_y': attack_feature_indices_y
    }
    # Calculate bounds for features to be used in clamping
    feature_cols = x_feature_cols.copy()
    feature_cols.extend(y_feature_cols)
    feature_mins, feature_maxs = get_feature_bounds(root_dir, station_ids, training_dates, feature_cols)

    # Prepare arguments for parallel processing
    process_args = [
        (station_id, root_dir, output_dir, training_dates, feature_mins, feature_maxs, attack_config, static_df)
        for station_id in station_ids
    ]
    # Process stations in parallel
    for args in tqdm(process_args):
        process_station(args)

if __name__ == "__main__":
    root_dir = "../climate_new"
    output_dir = "../climate_new_pgd_0.2x_0.1_0.025_50"
    model_path = "/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax_y/baseline_dropout_0.3/models/lstm/1016_145644/best_target_state_dict.pth"
    config_path = "/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax_y/baseline_dropout_0.3/models/lstm/1016_145644/config.json"

    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('-c', '--config', default=config_path, type=str,
                      help='config file path (default: config_LSTM.json)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: none)')
    config = ConfigParser.from_args(args)
    model = LSTM(**config['arch'])
    state_dicts = torch.load(model_path)
    model.load_state_dict(state_dicts[0])
    model = model.cuda()

    data_class = getattr(dataset, config['dataset'])
    train_data = data_class("../climate_new", config['normalize'], split="train",
                            x_feature=config['x_feature'], y_feature=config['y_feature'], exclude=config['exclude'],
                            minmax_feature=config['minmax_feature'])
    stats = train_data.get_stats()
    x_transform = train_data.get_x_transform()
    x_detransform = train_data.get_x_detransform()
    y_transform = train_data.get_target_transform()
    y_detransform = train_data.get_target_detransform()

    
    # Perform PGD attack on x features
    main(
        root_dir, 
        output_dir, 
        epsilon=0.1,
        alpha=0.025,
        num_iterations=50, 
        attack_x=True,
        attack_y=False,
        attack_x_features=[
            'runoff', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr', 'ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4', 'distNTN', 'LAI', 'FAPAR', 'NPP'
        ]
    )
    # perform PGD attack on y features
    # main(
    #     root_dir, 
    #     output_dir, 
    #     epsilon=0.05, 
    #     alpha=0.0125,
    #     num_iterations=50, 
    #     attack_x=False, 
    #     attack_y=True, 
    #     attack_y_features=[
    #         "00010", "00095", "00300", "00400", "00405", "00600", "00605", "00618", 
    #         "00660", "00665", "00681", "00915", "00925", "00930", "00935", "00940", 
    #         "00945", "00955", "71846", "80154"
    #     ]
    # )

    # Attack specific features
    # main(root_dir=root_dir, output_dir=output_dir, model_path=model_path,
    #      epsilon=0.1, alpha=0.01, num_steps=40, attack_x=True, attack_y=False,
    #      attack_specific_features=True, attack_feature_list=['runoff', 'pr'])

    # Attack both X and Y features
    # main(root_dir=root_dir, output_dir=output_dir, model_path=model_path,
    #      epsilon=0.1, alpha=0.01, num_steps=40, attack_x=True, attack_y=True,
    #      attack_specific_features=False, attack_feature_list=[])