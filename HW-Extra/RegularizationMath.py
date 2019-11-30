import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


#import training dataset
train_df = pd.read_csv('student-mat.csv', sep=';')

#see the columns in our data
train_df.info()
from sklearn.preprocessing import LabelEncoder

for column in train_df.columns:
    if train_df[column].dtype == type(object):
        le = LabelEncoder()
        train_df[column] = le.fit_transform(train_df[column])

# take a look at the head of the dataset
print(train_df.head())

#create our X and y
X = train_df.drop('G3', axis=1)
y = train_df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Training score: {}'.format(lr_model.score(X_train, y_train)))
print('Test score: {}'.format(lr_model.score(X_test, y_test)))

y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print('RMSE: {}'.format(rmse))


steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))