import pandas as pd
#%%
dataset_red=pd.read_csv("C:/Users/10253/Desktop/Machine-Learning-Project-Group-6/Data/红酒白酒分类+回归（都可以）/winequality-red.csv",sep=";",index_col=False)
dataset_white=pd.read_csv("C:/Users/10253/Desktop/Machine-Learning-Project-Group-6/Data/红酒白酒分类+回归（都可以）/winequality-white.csv",sep=";",index_col=False)

#%%
# dataset_red.head(n=len(dataset_red))
red=["red"]*len(dataset_red)
dataset_red.insert(0,"Label",red)
white=["white"]*len(dataset_white)
dataset_white.insert(0,"Label",white)
# dataset_white.head(n=5)
dataset=pd.concat([dataset_red,dataset_white],axis=0)
dataset.set_index("Label")
#%%
X=dataset.drop("Label",axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
Xs=scaler.fit_transform(X)
#%%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["Label"]=le.fit_transform(dataset["Label"])

y=dataset["Label"]
#%%
from sklearn.feature_selection import SelectKBest,chi2
X=SelectKBest(chi2,k=3).fit_transform(X,y)
#%%
from sklearn.model_selection import train_test_split

Xs_train,Xs_test,y_train,y_test=train_test_split(Xs,y,test_size=0.3,random_state=1,stratify=y)


#%%
from sklearn.svm import SVC

clf=SVC(C=1.0,kernel="rbf",degree=3,gamma='auto',probability=True)
clf.fit(Xs_train,y_train)
#%%

#%%
classifier_score=clf.score(Xs_test,y_test)
print('The classifier accuracy score is {:03.2f}'.format(classifier_score))

#%%
from sklearn.metrics import accuracy_score
y_predict=clf.predict(Xs_test)
accuracy=accuracy_score(y_test,y_predict)

print(accuracy)