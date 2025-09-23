import pandas as pd

def compare_csv(file1, file2):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 1. Check if all columns in df1 are in df2 (df1 columns is a subset of df2)
    if not set(df1.columns).issubset(set(df2.columns)):
        raise ValueError("Not all columns in the first dataframe (df1) are present in the second dataframe (df2).")
    
    # 2. Ensure 'staid' columns exist in both DataFrames
    if 'staid' not in df1.columns or 'staid' not in df2.columns:
        raise ValueError("The 'staid' column must be present in both dataframes.")

    # Check if 'staid' fields are identical
    if not df1['staid'].equals(df2['staid']):
        print("The 'staid' fields are not identical.")
        return False

    mismatches = []

    # Iterate through each column in df1
    for col in df1.columns:
        # Iterate through each row by index
        for i in range(len(df1)):
            val1, val2 = df1.at[i, col], df2.at[i, col]

            # Check if both values are NaN or both have valid values
            if (pd.isnull(val1) and not pd.isnull(val2)) or (not pd.isnull(val1) and pd.isnull(val2)):
                # Record discrepancies for NaN
                mismatches.append(f"Discrepancy in column '{col}', row {i+1} ('staid'={df1.at[i, 'staid']}): {val1} vs {val2}")
            assert (pd.isnull(val1) == pd.isnull(val2) ) or (not pd.isnull(val1) and not pd.isnull(val2))

            if col.endswith('_num'):
                print(f"col: {col}, val1: {val1}, val2: {val2}")
                # Convert both values to integers if they are not NaN
                if not pd.isnull(val1) and not pd.isnull(val2):
                    try:
                        int_val1, int_val2 = int(val1), int(val2)
                        if int_val1 != int_val2:
                            mismatches.append(f"Integer mismatch in column '{col}', row {i+1} ('staid'={df1.at[i, 'staid']}): {int_val1} vs {int_val2}")
                    except ValueError:
                        mismatches.append(f"Non-integer value found in column '{col}', row {i+1} ('staid'={df1.at[i, 'staid']}): {val1} vs {val2}")

    # Print all mismatches if there are any
    if mismatches:
        print("Mismatches found:")
        for mismatch in mismatches:
            print(mismatch)
        return False
    else:
        return True  # If no mismatches

# file_a = '/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax/baseline/models/lstm/1007_204343/predictions/metric/metrics.csv'
file_a = '/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax/baseline/models/lstm/1008_021147/predictions/metric/metrics.csv'
# file_a = '/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_twostatic_threedate_logminmax/baseline/models/Mnist_LeNet/1007_205006/predictions/metric/metric.csv'
file_b = '/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_twostatic_threedate/baseline/models/Mnist_LeNet/0828_141906/predictions/metric/metric.csv'
result = compare_csv(file_a, file_b)
print("Comparison result:", result)