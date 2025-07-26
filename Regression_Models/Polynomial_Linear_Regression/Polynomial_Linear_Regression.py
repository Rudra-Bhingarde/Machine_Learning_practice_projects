import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:,1:-1].values
Y = ds.iloc[:,-1].values

#simple linear regression

from sklearn.linear_model import LinearRegression
regressor_1 = LinearRegression()
regressor_1.fit(X,Y)
plt.scatter(X,Y,color="red")
plt.plot(X,regressor_1.predict(X),color='blue')
plt.title("simple linear regerssion")
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=9)
X_poly=poly.fit_transform(X)
regressor_2 = LinearRegression()
regressor_2.fit(X_poly,Y)
plt.scatter(X,Y,color="red")
plt.plot(X,regressor_2.predict(X_poly),color='blue')
plt.title("simple linear regerssion")
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
