import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

data = pd.read_csv('bc.csv');
del data['Id']
del data['Unnamed: 0']
# print(data)

data['Class'] = data['Class'].map({'benign': 0, 'malignant': 1})
# y=data['Class']
# del data['Class']
# x = data
train, test = train_test_split(data, test_size=0.2)
# X_train, X_test, Y_train, Y_test = train_test_split(data, data['Class'], test_size=0.2)
# logmodel = LogisticRegression()
# logmodel.fit(X_train, Y_train)
# predictions = logmodel.predict(X_test)
# print (classification_report(Y_test, predictions))
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logmodel.score(X_test, Y_test)))
# df_majority = data[data.Class==0]

# model = sm.formula.glm("Class ~ C(Cell.shape)", family = sm.families.Binomial(), data=data).fit()
df_majority = data[data.Class==0]
df_minority = data[data.Class==1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=239, random_state=123)

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

print(df_downsampled.Class.value_counts())
print((df_downsampled))

train, test = train_test_split(df_downsampled, test_size=0.3)

y = train['Class']
x = train.drop('Class', axis=1)
print(x)
x = train[['Cl.thickness', 'Cell.size', 'Cell.shape']]

glm_model = sm.GLM(y, x, family=sm.families.Binomial())
result = glm_model.fit()
print(result.summary())
x_test = test[['Cl.thickness', 'Cell.size', 'Cell.shape']]

#
predict = result.predict(x_test)
y_prediction = [1 if x > 0.5 else 0 for x in predict]
print(y_prediction)
print( accuracy_score(y_prediction, test['Class']) )
# print(model.summary())
