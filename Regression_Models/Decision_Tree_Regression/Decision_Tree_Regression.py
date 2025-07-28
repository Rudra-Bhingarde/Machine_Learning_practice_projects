import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:,1:-1].values
Y = ds.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

print(regressor.predict([[6.5]]))

xgrid = np.arange(min(X),max(X),0.1)
print("xgrid=",xgrid)
xgrid = xgrid.reshape(len(xgrid),1)
print("xgrid=",xgrid)
plt.scatter(X,Y,color='red')
plt.plot(xgrid,regressor.predict(xgrid),color="blue")
plt.title('decisiontree regression ')
plt.xlabel('position')
plt.ylabel('salaries')
plt.show()