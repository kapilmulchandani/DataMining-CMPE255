from sklearn.model_selection import train_test_split
import pandas as pd
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import statsmodels.formula.api as smf

#Prepocessing
data = pd.read_csv('bc.csv');
del data['Id']
del data['Unnamed: 0']
data = data.rename(columns = {"Cl.thickness": "thickness"})
data = data.rename(columns = {"Cell.size": "size"})
data = data.rename(columns = {"Cell.shape": "shape"})

#Train-Test-Split
data['Class'] = data['Class'].map({'benign': 0, 'malignant': 1})
train, test = train_test_split(data, test_size=0.2)

# Downscaling
df_majority = train[train.Class==0]
df_minority = train[train.Class==1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=199)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

#Modelling
formula = 'Class ~ thickness+size+shape'
model = smf.glm(formula = formula, data=df_downsampled, family=sm.families.Binomial())
result = model.fit()

#Prediction
predict = result.predict(test)
y_prediction = [1 if x > 0.5 else 0 for x in predict]
print( accuracy_score(y_prediction, test['Class']) )
