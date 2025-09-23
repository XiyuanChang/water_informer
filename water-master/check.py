import pandas as pd

def compare_csv(file1, file2):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if 'staid' fields are identical
    if not df1['staid'].equals(df2['staid']):
        print("The 'staid' fields are not identical.")
        return False

    # Check if the columns are identical
    if set(df1.columns) != set(df2.columns):
        print("The columns are not identical.")
        return False

    mismatches = []
    for col in df1.columns:
        for i in range(len(df1[col])):
            val1, val2 = df1.at[i, col], df2.at[i, col]

            # Check for empty values or float comparison
            if pd.isnull(val1) and pd.isnull(val2):
                continue  # Both are NaN, thus equivalent for this context
            elif pd.isnull(val1) or pd.isnull(val2):
                mismatches.append(f"Empty value mismatch at column {col}, row {i+1}: {val1} vs {val2}")

    # Print all mismatches if there are any
    if mismatches:
        print("Mismatches found:")
        for mismatch in mismatches:
            print(mismatch)
        return False
    else:
        return True  # If no mismatches

# Example usage
# wrtds vs deeponet
# stationA
# file_a = '/home/dev01/deeponet/MyProject/wrtdsA_metric.csv'
# file_b = '/home/dev01/deeponet/MyProject/saved_deepOnetA/models/Mnist_LeNet/0522_201011/predictions/metric/metric.csv'
# file_b = '/home/dev01/deeponet/MyProject/saved_stationA/models/lstm/0523_135215/predictions/metric/metric_testt.csv' # LSTM
# stationB
file_a = '/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_lstm_logminmax/baseline/models/lstm/1007_204343/predictions/metric/metrics.csv'
file_b = '/home/kh31/Xiaobo/deeponet/MyProject/unified_ablation_twostatic_threedate_logminmax/baseline/models/Mnist_LeNet/1007_205006/predictions/metric/metric.csv' 
# file_b = '/home/dev01/deeponet/MyProject/saved_stationB/models/lstm/0523_135321/predictions/metric/metric_testt.csv' # LSTM
# stationC
# file_a = '/home/dev01/deeponet/MyProject/wrtdsC_metric.csv'
# file_b = '/home/dev01/deeponet/MyProject/saved_deepOnetC/models/Mnist_LeNet/0522_201530/predictions/metric/metric.csv'
# file_b = '/home/dev01/deeponet/MyProject/saved_stationC/models/lstm/0523_135915/predictions/metric/metric_testt.csv'# LSTM

result = compare_csv(file_a, file_b)
print("Comparison result:", result)

