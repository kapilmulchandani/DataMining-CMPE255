from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np

dtData = pd.read_csv('DecisionTree.csv')
print(dtData)

dtData['Windy'] = dtData['Windy'].map({'No': 0, 'Yes': 1})
dtData['Air Quality Good'] = dtData['Air Quality Good'].map({'No': 0, 'Yes': 1})
dtData['Hot'] = dtData['Hot'].map({'No': 0, 'Yes': 1})
dtData['Play Tennis'] = dtData['Play Tennis'].map({'No': 0, 'Yes': 1})

X = dtData.drop(['Play Tennis'], axis=1)
Y = dtData['Play Tennis']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

decisionTree = DecisionTreeClassifier()
decisionTree.fit(X_train, Y_train)

# (graph, ) = graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

y_pred = decisionTree.predict(X_test)

dot_data = StringIO()
export_graphviz(decisionTree, out_file=dot_data)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_pdf("iris.pdf")

species = np.array(Y_test)
predictions = np.array(y_pred)
