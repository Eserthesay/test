#import libs

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import pandas as pd

house=fetch_california_housing()

X=pd.DataFrame(house.data,columns=house.feature_names)
y=house.target

reg=LinearRegression()

model=reg.fit(X,y)
