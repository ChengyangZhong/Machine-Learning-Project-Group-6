import pandas as pd
import numpy as np

# dataset_red=pd.read_csv("Data/winequality-red.csv",sep=";",index_col=False)
# dataset_white=pd.read_csv("Data/winequality-white.csv",sep=";",index_col=False)
dataset_red=pd.read_csv("C:/Users/10253/Desktop/Machine-Learning-Project-Group-6/Data/winequality-red.csv",sep=";",index_col=False)
dataset_white=pd.read_csv("C:/Users/10253/Desktop/Machine-Learning-Project-Group-6/Data/winequality-white.csv",sep=";",index_col=False)

red=["red"]*len(dataset_red)
dataset_red.insert(0,"Label",red)
white=["white"]*len(dataset_white)
dataset_white.insert(0,"Label",white)
# dataset_white.head(n=5)
dataset=pd.concat([dataset_red,dataset_white],axis=0)
dataset.set_index("Label")

from matplotlib import pyplot as plt
import seaborn as sns
fig,axes=plt.subplots(nrows=4,ncols=3,figsize=(15,20))
fig.subplots_adjust(hspace=0.2,wspace=.5)
axes=axes.ravel()

for i,col in enumerate(dataset.columns[1:]):
    _=sns.boxplot(y=col,x='Label',data=dataset,ax=axes[i])


corrMatt = dataset.corr()
    
mask = np.zeros_like(corrMatt)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(20, 12))
plt.title('Breast Cancer Feature Correlation')

cmap = sns.diverging_palette(260, 10, as_cmap=True)

sns.heatmap(corrMatt, vmax=1.2, square=False, cmap=cmap, mask=mask,
ax=ax, annot=True, fmt='.2g', linewidths=1);    


X=dataset.drop("Label",axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
Xs=scaler.fit_transform(X)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["Label"]=le.fit_transform(dataset["Label"])

y=dataset["Label"]


from sklearn.feature_selection import SelectKBest,f_classif
Xs=SelectKBest(f_classif,k=4).fit_transform(Xs,y)


from sklearn.model_selection import train_test_split

Xs_train,Xs_test,y_train,y_test=train_test_split(Xs,y,test_size=0.3,random_state=1,stratify=y)


"""
  DecisionTreeClassifier
"""
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.tree import DecisionTreeClassifier

clf_fs_cv=Pipeline(
    [('feature selector',SelectKBest(f_classif,k=4)),('decision trees',DecisionTreeClassifier(criterion="entropy",splitter="best"))]
)
score=cross_val_score(clf_fs_cv,Xs,y,cv=10)
print(score.mean())


"""
  DecisionTreeRegressor
"""

# dataset=pd.read_csv('/Data/airfoil_self_noise.dat',sep="\t",engine='python',
#                     names=["Frequency","Angle","Chord","Velocity","Suction","Pressure Level"],header=None)
dataset=pd.read_csv('C:/Users/10253/Desktop/Machine-Learning-Project-Group-6/Data/airfoil_self_noise.dat',sep="\t",engine='python',
                    names=["Frequency","Angle","Chord","Velocity","Suction","Pressure Level"],header=None)

Xs=dataset.drop("Pressure Level",axis=1)
y=dataset["Pressure Level"]

from sklearn.tree import DecisionTreeRegressor
Xs_train,Xs_test,y_train,y_test=train_test_split(Xs,y)
regressor=Pipeline(
    [('decision trees',DecisionTreeRegressor(random_state=0))]
)
regressor.fit(Xs_train,y_train)
score=cross_val_score(regressor,Xs_train,y_train,cv=10,scoring="neg_mean_squared_error")
print(score.mean())
print(regressor.score(Xs_test,y_test))

"""
  ANN: Sequential for regression
"""
from tensorflow import keras
from sklearn.preprocessing import Normalizer

Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=1)

normal=Normalizer()
Xs_train=normal.fit_transform(Xs_train)

Xs_test=normal.transform(Xs_test)

# y_train=normal.fit_transform(y_train.values.reshape(-1,1))
# y_test=normal.transform(y_test.values.reshape(-1,1))


model=keras.models.Sequential()
model.add(keras.layers.Dense(512,input_dim=len(dataset.columns)-1,activation="relu",use_bias=True))
model.add(keras.layers.Dense(210,activation="relu",use_bias=True))
model.add(keras.layers.Dense(320,activation="relu",use_bias=True))
model.add(keras.layers.Dense(1,use_bias=True))

model.compile(loss="mse",optimizer="adam")

history=model.fit(np.array(Xs_train),np.array(y_train),epochs=20,validation_data=(np.array(Xs_test),np.array(y_test)),verbose=1,batch_size=128)
score=model.evaluate(np.array(Xs_test),np.array(y_test),verbose=0)
# y_pred=model.predict(np.array(Xs_test))
print(score)

