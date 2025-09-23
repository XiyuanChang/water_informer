import pandas as pd

a = pd.read_csv('/home/kh31/Xiaobo/deeponet/climate_new/static_filtered.csv')

col = a.columns.tolist()
cols = col[-1:] + col[:-1]
a = a[cols]
a.to_csv('/home/kh31/Xiaobo/deeponet/climate_new/static_filtered.csv', index=False)

