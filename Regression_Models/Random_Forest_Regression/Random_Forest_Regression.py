import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv('Position_Salaries.csv')
X=ds.iloc[:,1:-1].values
Y = ds.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10000)
regressor.fit(X,Y)

xgrid = np.arange(np.min(X),np.max(X),0.1)
xgrid = xgrid.reshape(len(xgrid),1)
plt.scatter(X,Y,color='red')
plt.plot(xgrid,regressor.predict(xgrid),color='blue')
plt.title('random forest tree regression')
plt.xlabel('level')
plt.ylabel('salaries')
plt.show()
