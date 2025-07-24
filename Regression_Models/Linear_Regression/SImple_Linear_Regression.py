import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain,Ytrain)
Ypred = regressor.predict(Xtest)

plt.scatter(Xtrain,Ytrain,color="red")
plt.plot(Xtrain,regressor.predict(Xtrain),color="blue")
plt.title("years of experience vs salary")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

#evaluation
from sklearn.metrics import r2_score
print(r2_score(Ytest,Ypred))
