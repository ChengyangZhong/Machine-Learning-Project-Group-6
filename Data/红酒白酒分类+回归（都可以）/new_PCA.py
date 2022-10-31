# Author:Mingze Chen
# -*- codeing = utf-8 -*-
# @Time :31/10/2022 19:35
# @Author :empty
# @Site :
# @File :new_PCA.py
# @Software :PyCharm

import pandas as pd

dataset_red=pd.read_csv("winequality-red.csv",sep=";",index_col=False)
dataset_white=pd.read_csv("winequality-white.csv",sep=";",index_col=False)

# dataset_red.head(n=len(dataset_red))
red=["red"]*len(dataset_red)  #弄了n个red的array
dataset_red.insert(0,"Label",red)  #insert一个新的label在0的位置上，使用red数组
white=["white"]*len(dataset_white)
dataset_white.insert(0,"Label",white)
dataset=pd.concat([dataset_red,dataset_white],axis=0)  #将二者结合在一起

dataset.set_index("Label") #将label设为索引
X=dataset.drop("Label",axis=1)

# Label 红还是白
# X 没有label的结合体

#预处理
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
Xs=scaler.fit_transform(X)

# Xs 预处理过后的dataset

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["Label"]=le.fit_transform(dataset["Label"])

y = dataset["Label"]

# y 用0和1表示红和白

# get the number of features
features_number = len(dataset_red.columns)

from sklearn.decomposition import PCA
# use pca to modify the dataset
pca = PCA(n_components=features_number-1)

pca.fit(Xs, y)

# check
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

# SVM check

from sklearn.model_selection import train_test_split

Xs_train,Xs_test,y_train,y_test=train_test_split(Xs,y,test_size=0.3,random_state=1,stratify=y)

from sklearn.svm import SVC

clf=SVC(C=1.0,kernel="rbf",degree=3,gamma='auto',probability=True)
clf.fit(Xs_train,y_train)

#训练clf model

classifier_score=clf.score(Xs_test,y_test)

from sklearn.metrics import accuracy_score
y_predict=clf.predict(Xs_test)
accuracy=accuracy_score(y_test,y_predict)

print(accuracy)
