import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv('bc.csv');
del data['Id']
del data['Unnamed: 0']
print(data)

data['Class'] = data['Class'].map({'benign': 0, 'malignant': 1})
# train, test = train_test_split(data, test_size=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(data, data['Class'], test_size=0.2)
logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)
predictions = logmodel.predict(X_test)
print (classification_report(Y_test, predictions))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logmodel.score(X_test, Y_test)))
