import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds=pd.read_csv('50_Startups.csv')
X=ds.iloc[:,:-1].values
Y=ds.iloc[:,-1].values

#one hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=ct.fit_transform(X)


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=42)

print(Xtrain)
print(Ytrain)
print(Xtest)
print(Ytest)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(Xtrain,Ytrain)
Ypred=regressor.predict(Xtest)

np.set_printoptions(precision=2)
print(np.concatenate((Ypred.reshape(len(Ypred),1), Ytest.reshape(len(Ytest),1)),1))