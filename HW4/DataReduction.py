from sklearn.datasets import load_wine
import pandas as pd
import chart_studio.plotly as ply
import matplotlib.pylab as plt

wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
print(wine_df['proline'].max())
print(wine_df['proline'].min())
print(wine_df.shape[0])
width = ((wine_df['proline'].max() - wine_df['proline'].min())/wine_df.shape[0])
print(width)
labelsnew = [0, 1, 2, 3, 4]
wine_df['bin'] = pd.cut(wine_df['proline'], 5, labels=labelsnew)
sumOfBins = [0, 0, 0, 0, 0]
countOfBins = [0, 0, 0, 0, 0]
averageOfBins = [0, 0, 0, 0, 0]
print(wine_df['bin'][1])
for i in range(0, wine_df.shape[0]):
    sumOfBins[wine_df['bin'][i]] = sumOfBins[wine_df['bin'][i]] + wine_df['proline'][i]
    countOfBins[wine_df['bin'][i]] = countOfBins[wine_df['bin'][i]] + 1;

for i  in range(0, len(sumOfBins)):
    averageOfBins[i] = sumOfBins[i]/countOfBins[i]

for i in range(0, wine_df.shape[0]):
    wine_df['proline'][i] = averageOfBins[wine_df['bin'][i]]
