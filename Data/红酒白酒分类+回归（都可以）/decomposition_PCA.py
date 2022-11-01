# Author:Mingze Chen
# -*- codeing = utf-8 -*-
# @Time :25/10/2022 09:58
# @Author :empty
# @Site :
# @File :decomposition_PCA.py
# @Software :PyCharm

from sklearn.decomposition import PCA
import pandas as pd

dataset_red=pd.read_csv("Data/红酒白酒分类+回归（都可以）/winequality-red.csv",sep=";",index_col=False)

# separating headers, x just like "header = None"
features = dataset_red.columns
x = dataset_red.loc[:, features].values

# 预处理
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

# get the number of features
features_number = len(dataset_red.columns)

# use pca to modify the dataset
pca = PCA(n_components=features_number-1)
pca.fit(dataset_red)

# check
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

# transformation
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)

print(principalDf.head(n = 5))