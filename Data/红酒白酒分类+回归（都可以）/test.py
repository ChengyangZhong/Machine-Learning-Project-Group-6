import pandas as pd
dataset=pd.read_csv("winequality-red.csv",sep=";",index_col=False)
dataset.head()