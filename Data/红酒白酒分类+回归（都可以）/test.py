import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

dataset_red=pd.read_csv("Data/红酒白酒分类+回归（都可以）/winequality-red.csv",sep=";",index_col=False)
dataset_white=pd.read_csv("Data/红酒白酒分类+回归（都可以）/winequality-white.csv",sep=";",index_col=False)


red=["red"]*len(dataset_red)
dataset_red.insert(0,"Label",red)
white=["white"]*len(dataset_white)
dataset_white.insert(0,"Label",white)
# dataset_white.head(n=5)
dataset=pd.concat([dataset_red,dataset_white],axis=0)
dataset.set_index("Label")

X=dataset.drop("Label",axis=1)

# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# Xs=scaler.fit_transform(X)
#
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# dataset["Label"]=le.fit_transform(dataset["Label"])
Xs=X

y=dataset["Label"]


"""
    Decision tree for classification and regression
"""
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.tree import DecisionTreeClassifier
X_train,X_test,y_train,y_test=train_test_split(Xs,y,test_size=0.3)

clf_fs_cv=Pipeline(
    [('feature selector',SelectKBest(f_classif,k=4)),('decision trees',DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=4))]
)
print(cross_val_score(clf_fs_cv,X_train,y_train,cv=10))

clf_fs_cv.fit(X_train,y_train)

print(classification_report(y_test,clf_fs_cv.predict(X_test)))
print(confusion_matrix(y_test,clf_fs_cv.predict(X_test)))


dataset=pd.read_csv('Data/加速度计 regression/airfoil_self_noise.dat',sep="\t",engine='python',names=["Frequency","Angle","Chord","Velocity","Suction","Pressure Level"],header=None)


X=dataset.drop("Pressure Level",axis=1)

from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# Xs=scaler.fit_transform(X)
Xs=X


from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# dataset["Pressure Level"]=le.fit_transform(dataset["Pressure Level"])

y=dataset["Pressure Level"]
# y=y[:1000]


import sklearn
from sklearn import tree
from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeRegressor

regressor=Pipeline(
    [('decision trees',DecisionTreeRegressor(random_state=0,max_depth=8,max_features=5,))]
)
Xs_train,Xs_test,y_train,y_test=train_test_split(Xs,y)
score=cross_val_score(regressor,Xs_train,y_train,cv=10,scoring="neg_mean_squared_error")
print(score.mean())
regressor.fit(Xs_train,y_train)
print(regressor.score(Xs_test,y_test))

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
tree.plot_tree(regressor[0],feature_names=list(Xs_train.columns))
plt.show()



# text_repre=tree.export_text(regressor['decision trees'])
# print(text_repre)