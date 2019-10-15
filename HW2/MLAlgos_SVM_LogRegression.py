from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Get Data
adultCensusdata = pd.read_csv('adult.csv')

# Convert to Numeric from Strings
labelEncoder = preprocessing.LabelEncoder()
adultCensusdata['workclass'] = labelEncoder.fit_transform(adultCensusdata['workclass'])
adultCensusdata['education'] = labelEncoder.fit_transform(adultCensusdata['education'])
adultCensusdata['marital.status'] = labelEncoder.fit_transform(adultCensusdata['marital.status'])
adultCensusdata['occupation'] = labelEncoder.fit_transform(adultCensusdata['occupation'])
adultCensusdata['relationship'] = labelEncoder.fit_transform(adultCensusdata['relationship'])
adultCensusdata['race'] = labelEncoder.fit_transform(adultCensusdata['race'])
adultCensusdata['sex'] = labelEncoder.fit_transform(adultCensusdata['sex'])
adultCensusdata['native.country'] = labelEncoder.fit_transform(adultCensusdata['native.country'])
adultCensusdata['income'] = labelEncoder.fit_transform(adultCensusdata['income'])

# Features and Classes
X = adultCensusdata.drop(['income'], axis=1)
Y = adultCensusdata['income']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# Normalization
X_train_min = X_train.min()
X_train_max = X_train.max()
X_train_range = (X_train_max - X_train_min)
X_train_scaled =  (X_train - X_train_min)/X_train_range

X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range

# SVM Model
svc_model = SVC()
svc_model.fit(X_train, Y_train)
predict = svc_model.predict(X_test)
cm = np.array(confusion_matrix(Y_test, predict, labels=[0, 1]))
print("SVM Model")
print(cm)
print( 'Accuracy: ' ,accuracy_score(predict, Y_test)*100)
print('Precision: ', precision_score(predict, Y_test)*100)
print('Classification Report: ', classification_report(predict, Y_test))
print()
print('*************************************')

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)
predict_logistic = logistic_model.predict(X_test)
cm_logistic = np.array(confusion_matrix(Y_test, predict_logistic, labels=[0,1]))
print("Logistic Regression Model")
print(cm_logistic)
print( 'Accuracy: ' ,accuracy_score(predict_logistic, Y_test)*100)
print('Precision: ', precision_score(predict_logistic, Y_test)*100)
print('Classification Report: ', classification_report(predict_logistic, Y_test))