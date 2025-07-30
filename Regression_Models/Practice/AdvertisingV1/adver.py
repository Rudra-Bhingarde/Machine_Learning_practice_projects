import pandas as pd
import numpy as np

ds = pd.read_csv('advertising.csv')
X = ds.iloc[:,:-1].values
Y = ds.iloc[:,-1].values

#dealing with missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy = "mean")
X=imputer.fit_transform(X)
print(X)

#splitting data in training and test
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=42)

#training the model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(Xtrain,Ytrain)
Ypred = regressor.predict(Xtest)
print(Ypred)
print(Ytest)
print(np.concatenate((Ypred.reshape(len(Ypred),1), Ytest.reshape(len(Ytest),1)),1))
from sklearn.metrics import r2_score
print(r2_score(Ytest,Ypred))