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
        train_mask = df.iloc[:, 0].isin(training_dates)
        df_train = df[train_mask]
        
        for col in feature_cols:
            valid_values = df_train[col].dropna()
            if len(valid_values) == 0:
                continue
                
            sorted_values = np.sort(valid_values.unique())
            
            if len(sorted_values) >= 2:
                curr_min = sorted_values[0]
                curr_max = sorted_values[-1]
                
                # Update global values
                if col not in feature_mins or curr_min < feature_mins[col]:
                    feature_mins[col] = curr_min
                if col not in feature_maxs or curr_max > feature_maxs[col]:
                    feature_maxs[col] = curr_max
                    
    return feature_mins, feature_maxs

def process_station(args):
    """Process a single station file. Designed for parallel processing."""
    station_id, root_dir, output_dir, training_dates, feature_bounds, outlier_config = args
    feature_mins, feature_maxs = feature_bounds
    
    input_path = os.path.join(root_dir, f"{station_id}.csv")
    output_path = os.path.join(output_dir, f"{station_id}.csv")
    
    if not os.path.exists(input_path):
        return

    # Read the data
    df = pd.read_csv(input_path)
    df.columns.values[0] = ''
    train_mask = df.iloc[:, 0].isin(training_dates)
    
    # Process X features if configured
    if outlier_config['add_x_outliers']:
        x_features = df.loc[train_mask, outlier_config['x_feature_cols']].to_numpy()
        valid_mask = ~np.isnan(x_features)
        valid_indices = np.where(valid_mask)
        x_features_copy = copy.deepcopy(x_features)
        
        if len(valid_indices[0]) > 0:
            num_to_outlier = int(len(valid_indices[0]) * outlier_config['percent_outliers'])
            random_indices = np.random.choice(len(valid_indices[0]), num_to_outlier, replace=False)
            rows_to_change = valid_indices[0][random_indices]
            cols_to_change = valid_indices[1][random_indices]
            
            for i, (r, c) in enumerate(zip(rows_to_change, cols_to_change)):
                col_name = outlier_config['x_feature_cols'][c]
                current_value = x_features[r, c]
                
                # Determine if this should be a high or low outlier
                make_high = np.random.random() < outlier_config['high_outlier_ratio']
                
                if make_high:
                    # Skip if current value is already the maximum
                    if current_value != feature_maxs[col_name]:
                        x_features[r, c] = feature_maxs[col_name]
                else:
                    # Skip if current value is already the minimum
                    if current_value != feature_mins[col_name]:
                        x_features[r, c] = feature_mins[col_name]
            
            ratio_x = (np.sum(x_features!=x_features_copy) - np.sum(np.isnan(x_features))) / len(valid_indices[0])
            print(f"Station {station_id} - ratio x outliers:", ratio_x)
            df.loc[train_mask, outlier_config['x_feature_cols']] = x_features
    
    # Process Y features if configured
    if outlier_config['add_y_outliers']:
        y_features = df.loc[train_mask, outlier_config['y_feature_cols']].to_numpy()
        valid_mask = ~np.isnan(y_features)
        valid_indices = np.where(valid_mask)
        y_features_copy = copy.deepcopy(y_features)
        
        if len(valid_indices[0]) > 0:
            num_to_outlier = int(len(valid_indices[0]) * outlier_config['percent_outliers'])
            random_indices = np.random.choice(len(valid_indices[0]), num_to_outlier, replace=False)
            rows_to_change = valid_indices[0][random_indices]
            cols_to_change = valid_indices[1][random_indices]
            
            for i, (r, c) in enumerate(zip(rows_to_change, cols_to_change)):
                col_name = outlier_config['y_feature_cols'][c]
                current_value = y_features[r, c]
                
                # Determine if this should be a high or low outlier
                make_high = np.random.random() < outlier_config['high_outlier_ratio']
                
                if make_high:
                    # Skip if current value is already the maximum
                    if current_value != feature_maxs[col_name]:
                        # Make it slightly higher than the current maximum
                        y_features[r, c] = feature_maxs[col_name]
                else:
                    # Skip if current value is already the minimum
                    if current_value != feature_mins[col_name]:
                        # Make it slightly lower than the current minimum
                        y_features[r, c] = feature_mins[col_name]
            
            ratio_y = (np.sum(y_features!=y_features_copy) - np.sum(np.isnan(y_features))) / len(valid_indices[0])
            print(f"Station {station_id} - ratio y outliers:", ratio_y)
            df.loc[train_mask, outlier_config['y_feature_cols']] = y_features
    
    df.to_csv(output_path, index=False, na_rep="")


def main(root_dir="climate_new", output_dir="climate_new_outliers", percent_outliers=0.1, 
         high_outlier_ratio=0.5, add_x_outliers=True, add_y_outliers=False):
    """
    Add outliers to the dataset.
    
    Parameters:
    - root_dir: Source directory containing original data
    - output_dir: Target directory for modified data
    - percent_outliers: Percentage of valid values to convert to outliers
    - high_outlier_ratio: Ratio of outliers that should be high (vs low)
    - add_x_outliers: Whether to add outliers to X features
    - add_y_outliers: Whether to add outliers to Y features
    """
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
    
    # Define feature columns (using placeholders as specified)
    # x_feature_cols = ["runoff", "pr", "sph", "srad", "tmmn", "tmmx", "pet", "etr", "ph", "Conduc", 
    #                  "Ca", "Mg", "K", "Na", "NH4", "NO3", "Cl", "SO4", "distNTN", "LAI", "FAPAR", 
    #                 "NPP"]
    x_feature_cols = ["runoff"]
    y_feature_cols = ["00010", "00095", "00300", "00400", "00405", "00600", "00605", "00618", 
                     "00660", "00665", "00681", "00915", "00925", "00930", "00935", "00940", 
                     "00945", "00955", "71846", "80154"]
    
    # Configure outlier parameters
    outlier_config = {
        'percent_outliers': percent_outliers,
        'high_outlier_ratio': high_outlier_ratio,
        'add_x_outliers': add_x_outliers,
        'add_y_outliers': add_y_outliers,
        'x_feature_cols': x_feature_cols,
        'y_feature_cols': y_feature_cols
    }
    
    # Calculate bounds for all features that will be modified
    feature_cols = []
    if add_x_outliers:
        feature_cols.extend(x_feature_cols)
    if add_y_outliers:
        feature_cols.extend(y_feature_cols)
    
    print("Calculating feature bounds...")
    feature_bounds = get_feature_bounds(root_dir, station_ids, training_dates, feature_cols)
    
    # Prepare arguments for parallel processing
    process_args = [
        (station_id, root_dir, output_dir, training_dates, feature_bounds, outlier_config)
        for station_id in station_ids
    ]
    
    # Process stations in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()-5) as executor:
        list(tqdm(executor.map(process_station, process_args), total=len(station_ids)))

if __name__ == "__main__":
    root_dir = "climate_new"
    output_dir = "climate_new_outliers_0.2_high0_q"
    
    np.random.seed(42)
    random.seed(42)
    
    # Add 10% outliers to X features only, with 50% being high outliers
    # main(root_dir=root_dir, 
    #      output_dir=output_dir, 
    #      percent_outliers=0.1,
    #      high_outlier_ratio=1,  # x% high outliers, 1-x% low outliers
    #      add_x_outliers=True, 
    #      add_y_outliers=False)

    # Add 10% outliers to Y features only, with 50% being high outliers
    main(root_dir=root_dir, 
         output_dir=output_dir, 
         percent_outliers=0.2,
         high_outlier_ratio=0,  # x% high outliers, 1-x% low outliers
         add_x_outliers=True, 
         add_y_outliers=False)