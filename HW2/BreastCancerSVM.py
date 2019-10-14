from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC

# Getting Data
breastCancerdf = pd.read_csv('bc.csv')
X = breastCancerdf.drop(['Class'], axis=1)
Y = breastCancerdf['Class']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# Normalizing
X_train_min = X_train.min()
X_train_max = X_train.max()
X_train_range = (X_train_max - X_train_min)
X_train_scaled =  (X_train - X_train_min)/X_train_range

X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range

svc_model = SVC()
svc_model.fit(X_train_scaled, Y_train)

print("Support Vectors : \n",svc_model.support_vectors_)
print("Indices : \n", svc_model.support_)

print("Number of Support Vectors with Each Class : \n",svc_model.n_support_)
