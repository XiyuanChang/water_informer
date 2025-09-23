import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

num_clusters = 8
# path = "saved_deepOnetC_31/models/Mnist_LeNet/0625_143733/predictions/metric/metric.csv"
path = "saved_deepOnetC_105/models/Mnist_LeNet/0702_141400/predictions/metric/metric.csv"

data = pd.read_csv(path)
# y = data[["00600_r2", "00665_r2"]].values
col = ["00010", "00095", "00300", "00400", "00405", "00600", "00605", "00618", "00660", "00665", "00681", "00915", "00925", "00930", "00935", "00940", "00945", "00955", "71846", "80154"]
col = [c+"_r2" for c in col]
y = data[col].values

valid_idx = np.where(np.isnan(y).sum(axis=1) == 0)[0]
valid_y = y[valid_idx]
# print(data.iloc[valid_idx[np.any(valid_y < -100, axis=1)], [0, 21, 24, 37, 40]])
# print(valid_y[np.any(valid_y < -100, axis=1)])
# rule out the case where values are smaller than -100
# valid_idx = valid_idx[np.all(valid_y > -100, axis=1)]
# valid_y = valid_y[np.all(valid_y > -100, axis=1)]
# valid_y = valid_y[valid_idx]
print(valid_y.shape)
classifier = KMeans(n_clusters=num_clusters)
result = classifier.fit_predict(valid_y)
for i in range(num_clusters):
    print("Cluster %d: %d" % (i, np.sum(result == i)))

final_result = np.ones(len(y))*-1
final_result[valid_idx] = result
final_result_df = pd.DataFrame(final_result, columns=["cluster"])

# concat final result and data and save it to new csv
data = pd.concat([data, final_result_df], axis=1)
data.to_csv("saved_deepOnetC_105/models/Mnist_LeNet/0702_141400/predictions/metric/clustered_group8.csv", index=False)

# scatter plot clusters, use PCA to reduce dimension to 2
from sklearn.decomposition import PCA
from sklearn import manifold
import matplotlib.pyplot as plt
# pca = PCA(n_components=2)
# valid_y_pca = pca.fit_transform(valid_y)
# now use t-SNE to reduce dimension to 2
# valid_y_tsne = manifold.TSNE(n_components=2).fit_transform(valid_y)
# plt.scatter(valid_y_tsne[:, 0], valid_y_tsne[:, 1], c=result)
# plt.scatter(valid_y[:, 0], valid_y[:, 1], c=result)
# plt.savefig("cluster.png")
# print(final_result_df.head)