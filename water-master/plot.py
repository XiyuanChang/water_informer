import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

deeponet_c = pd.read_csv('unified_ablation_twostatic_threedate_logminmax_y/baseline1024/models/Mnist_LeNet/1018_003851/predictions/metric/metric.csv').values


lstm_c = pd.read_csv('unified_ablation_lstm_logminmax/baseline/models/lstm/1008_040613/predictions/metric/metrics.csv').values




deeponet_kge_index = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58]
deeponet_pbias_index = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
deeponet_r2_index = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]

lstm_kge_index = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58]
lstm_pbias_index = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
lstm_r2_index = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]

# deeponet

deeponet_c_kge, deeponet_c_r2, deeponet_c_pbias = deeponet_c[:, deeponet_kge_index], deeponet_c[:, deeponet_r2_index], deeponet_c[:, deeponet_pbias_index]

# lstm 
lstm_c_kge, lstm_c_r2, lstm_c_pbias = lstm_c[:, lstm_kge_index], lstm_c[:, lstm_r2_index], lstm_c[:, lstm_pbias_index]


# KGE
df1 = pd.DataFrame(deeponet_c_kge, columns=['Temp', 'Cond', 'DO', 'pH', 'CO2', 'TN', 'N-org', 'NO3', 'PO4', 'TP', 'NPOC', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'SiO2', 'NHx', 'SSC'])
df1['Dataset'] = 'DeepOnet'
df2 = pd.DataFrame(lstm_c_kge, columns=['Temp', 'Cond', 'DO', 'pH', 'CO2', 'TN', 'N-org', 'NO3', 'PO4', 'TP', 'NPOC', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'SiO2', 'NHx', 'SSC'])
df2['Dataset'] = 'LSTM'
# 合并两个DataFrame
df_combined = pd.concat([df1, df2])

# 将数据转为长格式
df_melted = pd.melt(df_combined, id_vars=['Dataset'], var_name='Group', value_name='Value')

# 绘制箱线图
plt.figure(figsize=(15, 10))
ax = sns.boxplot(x='Group', y='Value', hue='Dataset', data=df_melted, showfliers=False)
ax.set_title('C Setting')
ax.set_xlabel('Groups')
ax.set_ylabel('KGE')
plt.xticks(rotation=45)
plt.legend(title='Method')
plt.savefig('figures/KGE_C.png', bbox_inches='tight')
plt.show()

# R2

df1 = pd.DataFrame(deeponet_c_r2, columns=['Temp', 'Cond', 'DO', 'pH', 'CO2', 'TN', 'N-org', 'NO3', 'PO4', 'TP', 'NPOC', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'SiO2', 'NHx', 'SSC'])
df1['Dataset'] = 'DeepOnet'
df2 = pd.DataFrame(lstm_c_r2, columns=['Temp', 'Cond', 'DO', 'pH', 'CO2', 'TN', 'N-org', 'NO3', 'PO4', 'TP', 'NPOC', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'SiO2', 'NHx', 'SSC'])
df2['Dataset'] = 'LSTM'


# 合并两个DataFrame
df_combined = pd.concat([df1, df2])

# 将数据转为长格式
df_melted = pd.melt(df_combined, id_vars=['Dataset'], var_name='Group', value_name='Value')

# 绘制箱线图
plt.figure(figsize=(15, 10))
ax = sns.boxplot(x='Group', y='Value', hue='Dataset', data=df_melted, showfliers=False)
ax.set_title('C Setting')
ax.set_xlabel('Groups')
ax.set_ylabel('R2')
plt.xticks(rotation=45)
plt.legend(title='Method')
plt.savefig('figures/R2_C.png', bbox_inches='tight')
plt.show()

# PBIAS

df1 = pd.DataFrame(deeponet_c_pbias, columns=['Temp', 'Cond', 'DO', 'pH', 'CO2', 'TN', 'N-org', 'NO3', 'PO4', 'TP', 'NPOC', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'SiO2', 'NHx', 'SSC'])
df1['Dataset'] = 'DeepOnet'
df2 = pd.DataFrame(lstm_c_pbias, columns=['Temp', 'Cond', 'DO', 'pH', 'CO2', 'TN', 'N-org', 'NO3', 'PO4', 'TP', 'NPOC', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'SiO2', 'NHx', 'SSC'])
df2['Dataset'] = 'LSTM'

# 合并两个DataFrame
df_combined = pd.concat([df1, df2])

# 将数据转为长格式
df_melted = pd.melt(df_combined, id_vars=['Dataset'], var_name='Group', value_name='Value')

# 绘制箱线图
plt.figure(figsize=(15, 10))
ax = sns.boxplot(x='Group', y='Value', hue='Dataset', data=df_melted, showfliers=False)
ax.set_title('C Setting')
ax.set_xlabel('Groups')
ax.set_ylabel('PBIAS')
plt.xticks(rotation=45)
plt.legend(title='Method')
plt.savefig('figures/PBIAS_C.png', bbox_inches='tight')
plt.show()