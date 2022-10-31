import pandas as pd

dataset_red=pd.read_csv("winequality-red.csv",sep=";",index_col=False)
dataset_white=pd.read_csv("winequality-white.csv",sep=";",index_col=False)


# dataset_red.head(n=len(dataset_red))
red=["red"]*len(dataset_red)
dataset_red.insert(0,"Label",red)
white=["white"]*len(dataset_white)
dataset_white.insert(0,"Label",white)
# dataset_white.head(n=5)
dataset=pd.concat([dataset_red,dataset_white],axis=0)
dataset.set_index("Label")

X=dataset.drop("Label",axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
Xs=scaler.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["Label"]=le.fit_transform(dataset["Label"])

y=dataset["Label"]

# from sklearn.feature_selection import SelectKBest,  f_classif
#
# X=SelectKBest(f_classif,k=3).fit_transform(Xs,y)
#
# from sklearn.model_selection import train_test_split
#
# Xs_train,Xs_test,y_train,y_test=train_test_split(Xs,y,test_size=0.3,random_state=1,stratify=y)
#
#
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SelectFromModel
# from sklearn.svm import SVC
#
# clf=SVC(C=1.0,kernel="linear",degree=3,gamma='auto',probability=True)
# clf.fit(Xs_train,y_train)
# # clf=LinearSVC(C=0.01,penalty="l1",dual=False).fit(X,y)
# # model=SelectFromModel(clf,perfit=True)
#
#
# classifier_score=clf.score(Xs_test,y_test)
# print('The classifier accuracy score is {:03.2f}'.format(classifier_score))
#
# from sklearn.metrics import accuracy_score
# y_predict=clf.predict(Xs_test)
# accuracy=accuracy_score(y_test,y_predict)
#
# print(accuracy)


"""
    Decision tree for classification and regression
"""
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.tree import DecisionTreeClassifier

clf_fs_cv=Pipeline(
    [('feature selector',SelectKBest(f_classif,k=4)),('decision trees',DecisionTreeClassifier(criterion="entropy",splitter="best"))]
)
cross_val_score(clf_fs_cv,Xs,y,cv=10)
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
    [('decision trees',DecisionTreeRegressor(random_state=0))]
)
cross_val_score(regressor,Xs,y,cv=10,scoring="neg_mean_squared_error")

# text_repre=tree.export_text(regressor['decision trees'])
# print(text_repre)