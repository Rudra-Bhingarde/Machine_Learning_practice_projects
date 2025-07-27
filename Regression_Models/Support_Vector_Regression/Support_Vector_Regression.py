import numpy as np
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
ssx = StandardScaler()
Xs=ssx.fit_transform(X)

ssy = StandardScaler()
Ys=ssy.fit_transform(Y.reshape(len(Y),1))

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(Xs,Ys)
Ypred=regressor.predict(Xs)
import matplotlib.pyplot as plt
plt.scatter(X,Y.ravel(),color='red')
plt.plot(X,ssy.inverse_transform(Ypred.reshape(len(Ypred),1)))
plt.show()