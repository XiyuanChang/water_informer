import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor
import warnings
import random
import copy
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

def process_station(args):
    """Process a single station file. Designed for parallel processing."""
    station_id, root_dir, output_dir, training_dates, feature_mins, feature_maxs, noise_config = args
    
    input_path = os.path.join(root_dir, f"{station_id}.csv")
    output_path = os.path.join(output_dir, f"{station_id}.csv")
    
    if not os.path.exists(input_path):
        return
    
    # Read the data
    df = pd.read_csv(input_path)
    df.columns.values[0] = '' 
    train_mask = df.iloc[:, 0].isin(training_dates)
    
    # Process X features if configured
    if noise_config['add_x_noise']:
        x_features = df.loc[train_mask, noise_config['x_feature_cols']].to_numpy()
        valid_mask = ~np.isnan(x_features)
        valid_indices = np.where(valid_mask)
        
        x_features_copy = copy.deepcopy(x_features)
        if len(valid_indices[0]) > 0:
            
            num_to_perturb = int(len(valid_indices[0]) * noise_config['percent_to_perturb'])
            # Ensure even number for splitting
            num_to_perturb = (num_to_perturb // 2) * 2
            
            random_indices = np.random.choice(len(valid_indices[0]), num_to_perturb, replace=False)
            rows_to_perturb = valid_indices[0][random_indices]
            cols_to_perturb = valid_indices[1][random_indices]
            # print(cols_to_perturb)
            # Split indices into two equal groups
            half_point = num_to_perturb // 2
            
            # Create noise array: first half positive, second half negative
            noise = np.zeros(num_to_perturb)
            noise[:half_point] = noise_config['noise_range']  # Positive noise
            noise[half_point:] = -noise_config['noise_range']  # Negative noise
            
            # Shuffle the noise array to randomize positive/negative positions
            np.random.shuffle(noise)
            
            original_values = x_features[rows_to_perturb, cols_to_perturb]
            noisy_values = original_values * (1 + noise)
            
            for i, (r, c) in enumerate(zip(rows_to_perturb, cols_to_perturb)):
                col_name = noise_config['x_feature_cols'][c]
                noisy_values[i] = np.clip(noisy_values[i], feature_mins[col_name], feature_maxs[col_name])
            
            x_features[rows_to_perturb, cols_to_perturb] = noisy_values
            ratio_x = np.sum(x_features!=x_features_copy) / x_features_copy.size
            print("ratio x:", ratio_x)
            ratio_x = (np.sum(x_features!=x_features_copy) - np.sum(np.isnan(x_features))) / len(valid_indices[0])
            print("ratio x without NaN:", ratio_x)
            df.loc[train_mask, noise_config['x_feature_cols']] = x_features
    
    # Process Y features if configured
    if noise_config['add_y_noise']:
        y_features = df.loc[train_mask, noise_config['y_feature_cols']].to_numpy()
        valid_mask = ~np.isnan(y_features)
        valid_indices = np.where(valid_mask)
        y_features_copy = copy.deepcopy(y_features)
        if len(valid_indices[0]) > 0:
            num_to_perturb = int(len(valid_indices[0]) * noise_config['percent_to_perturb'])
            # Ensure even number for splitting
            num_to_perturb = (num_to_perturb // 2) * 2
            
            random_indices = np.random.choice(len(valid_indices[0]), num_to_perturb, replace=False)
            rows_to_perturb = valid_indices[0][random_indices]
            cols_to_perturb = valid_indices[1][random_indices]
            
            # Split indices into two equal groups
            half_point = num_to_perturb // 2
            
            # Create noise array: first half positive, second half negative
            noise = np.zeros(num_to_perturb)
            noise[:half_point] = noise_config['noise_range']  # Positive noise
            noise[half_point:] = -noise_config['noise_range']  # Negative noise
            
            # Shuffle the noise array to randomize positive/negative positions
            np.random.shuffle(noise)
            
            original_values = y_features[rows_to_perturb, cols_to_perturb]
            noisy_values = original_values * (1 + noise)
            
            for i, (r, c) in enumerate(zip(rows_to_perturb, cols_to_perturb)):
                col_name = noise_config['y_feature_cols'][c]
                noisy_values[i] = np.clip(noisy_values[i], feature_mins[col_name], feature_maxs[col_name])
            
            y_features[rows_to_perturb, cols_to_perturb] = noisy_values

            # calculate the actual ratio of perturbed values, excluding all the NaN values
            ratio_y = (np.sum(y_features!=y_features_copy) - np.sum(np.isnan(y_features))) / len(valid_indices[0])

            print("ratio y:", ratio_y)

            df.loc[train_mask, noise_config['y_feature_cols']] = y_features
    
    df.to_csv(output_path, index=False, na_rep="")

def main(root_dir="climate_new", output_dir="climate_new_noisy", percent_to_perturb=0.2, noise_range=0.1, add_x_noise=True, add_y_noise=False):

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
    static_df = pd.read_csv(os.path.join(root_dir, 'static_filtered.csv'), usecols=['STAID'])
    station_ids = static_df['STAID'].astype(str).tolist()
    station_ids = [id.zfill(8) for id in station_ids]
    
    # Define feature columns
    # x_feature_cols = ["runoff", "pr", "sph", "srad", "tmmn", "tmmx", "pet", "etr", "ph", "Conduc", 
    #                 "Ca", "Mg", "K", "Na", "NH4", "NO3", "Cl", "SO4", "distNTN", "LAI", "FAPAR", 
    #                 "NPP"]
    x_feature_cols = ["runoff"]
    
    y_feature_cols = ["00010", "00095", "00300", "00400", "00405", "00600", "00605", "00618", 
                     "00660", "00665", "00681", "00915", "00925", "00930", "00935", "00940", 
                     "00945", "00955", "71846", "80154"]
    
    # Configure noise parameters
    noise_config = {
        'percent_to_perturb': percent_to_perturb,
        'noise_range': noise_range,
        'add_x_noise': add_x_noise,
        'add_y_noise': add_y_noise,
        'x_feature_cols': x_feature_cols,
        'y_feature_cols': y_feature_cols
    }
    
    # Calculate bounds for all features that will be perturbed
    feature_cols = []
    if add_x_noise:
        feature_cols.extend(x_feature_cols)
    if add_y_noise:
        feature_cols.extend(y_feature_cols)
    
    print("Calculating feature bounds...")
    feature_mins, feature_maxs = get_feature_bounds(root_dir, station_ids, training_dates, feature_cols)
    
    # Prepare arguments for parallel processing
    process_args = [
        (station_id, root_dir, output_dir, training_dates, feature_mins, feature_maxs, noise_config)
        for station_id in station_ids
    ]
    
    # Process stations in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()-5) as executor:
        list(tqdm(executor.map(process_station, process_args), total=len(station_ids)))

if __name__ == "__main__":
    root_dir = "climate_new"
    output_dir = "climate_new_noisy_0.4_0.5_q"
    # output_dir = "climate_new_noisy_0.2_0.2_y"
    # Example usage with different configurations:
    # Add 20% noise to X features only (default behavior)
    # main()
    np.random.seed(42)
    random.seed(42)
    # Add 30% noise to both X and Y features with ±15% range
    # main(percent_to_perturb=0.3, noise_range=0.15, add_x_noise=True, add_y_noise=True)
    
    # Add 25% noise to Y features only with ±5% range
    main(root_dir, output_dir, percent_to_perturb=0.4, noise_range=0.5, add_x_noise=True, add_y_noise=False)
    # main(root_dir, output_dir, percent_to_perturb=0.2, noise_range=0.2, add_x_noise=False, add_y_noise=True)
    