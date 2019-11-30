import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import resample
breastCancerdata = pd.read_csv('bc.csv')
X = breastCancerdata.drop(['Class'], axis=1)
Y = breastCancerdata['Class']

breastCancerdata['Class'] = breastCancerdata['Class'].map({'benign': 0, 'malignant': 1})
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

df_majority = X_train[X_train.Class==0]
df_minority = X_train[X_train.Class==1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=199)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])



def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel