from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn import preprocessing

adultCensusdata = pd.read_csv('adult.csv')
#
# print("age ", adultCensusdata['age'].unique())
# print("workclass ", adultCensusdata['workclass'].unique())
# print("education ", adultCensusdata['education'].unique())
# print("marital.status ", adultCensusdata['marital.status'].unique())
# print("occupation ", adultCensusdata['occupation'].unique())
# print("relationship ", adultCensusdata['relationship'].unique())
# print("race ", adultCensusdata['race'].unique())
# print("sex ", adultCensusdata['sex'].unique())
# print("native.country ", adultCensusdata['native.country'].unique())

labelEncoder = preprocessing.LabelEncoder()
adultCensusdata['workclass'] = labelEncoder.fit_transform(adultCensusdata['workclass'])
adultCensusdata['education'] = labelEncoder.fit_transform(adultCensusdata['education'])
adultCensusdata['marital.status'] = labelEncoder.fit_transform(adultCensusdata['marital.status'])
adultCensusdata['occupation'] = labelEncoder.fit_transform(adultCensusdata['occupation'])
adultCensusdata['relationship'] = labelEncoder.fit_transform(adultCensusdata['relationship'])
adultCensusdata['race'] = labelEncoder.fit_transform(adultCensusdata['race'])
adultCensusdata['sex'] = labelEncoder.fit_transform(adultCensusdata['sex'])
adultCensusdata['native.country'] = labelEncoder.fit_transform(adultCensusdata['native.country'])

X = adultCensusdata.drop(['income'], axis=1)
Y = adultCensusdata['income']

encoded_values = labelEncoder.fit_transform(["<=50k", ">50k"])
# print(encoded_values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

X_train_min = X_train.min()
X_train_max = X_train.max()
X_train_range = (X_train_max - X_train_min)
X_train_scaled =  (X_train - X_train_min)/X_train_range

X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range

svc_model = SVC()
svc_model.fit(X_train, Y_train)

predict = svc_model.predict(X_test)
cm = np.array(confusion_matrix(Y_test, predict, labels=['0', '1']))
print(cm)