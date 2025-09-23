import pandas as pd

df = pd.read_csv('/home/kh31/Xiaobo/deeponet/MyProject/saved_lstmC_105/models/lstm/0712_150846/predictions/metric/metric_num.csv')

y = ["00010", "00095", "00300", "00400", "00405", "00600", "00605", "00618", "00660", "00665", "00681", "00915", "00925", "00930", "00935", "00940", "00945", "00955", "71846", "80154"]

columns = ["staid"]
for i in y:
    columns.append(i + "_kge")
    columns.append(i + "_pbias")
    columns.append(i + "_r2")
    columns.append(i + "_num")
df = df[columns]
df.to_csv('/home/kh31/Xiaobo/deeponet/MyProject/saved_lstmC_105/models/lstm/0712_150846/predictions/metric/metric_num1.csv', index=False, na_rep='')