import pandas as pd

df_station = pd.read_csv("climate_new/static_filtered.csv")

# Iterate through the 'STAID' field
for staid in df_station['STAID']:
    # if staid != '14241500':
    #     continue
    staid_str = str(staid)

    if len(staid_str) < 8:
        staid_str = '0' * (8 - len(staid_str)) + staid_str
    
    # Read the corresponding CSV file
    
    csv_file = f'climate_new/{staid_str}.csv'
    try:
        data = pd.read_csv(csv_file)
    except:
        print(f"Error reading file {csv_file}")
        continue
    data.rename(columns={data.columns[0]: 'Date'}, inplace=True)
    
    data = data.loc[~(data.iloc[:, 1:21].isna().all(axis=1)), :]
    data = data.loc[~(data.iloc[:, 21:].isna().any(axis=1)), :]
    # data = data.dropna(subset=df.columns[21:], how='any')
    new_file = f'climate_washed/{staid_str}.csv'
    # if staid == '14241500':
    #     print(data.shape)
    data.to_csv(new_file, index=False)
    print("Washed file saved to", new_file)