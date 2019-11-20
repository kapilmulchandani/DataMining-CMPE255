from sklearn.datasets import load_wine
import pandas as pd

wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
for key, value in wine_df.iteritems():
    width = ((value.max() - value.min())/wine_df.shape[0])
    labelsnew = [0, 1, 2, 3, 4]
    wine_df['bin'] = pd.cut(value, 5, labels=labelsnew)
    sumOfBins = [0, 0, 0, 0, 0]
    countOfBins = [0, 0, 0, 0, 0]
    averageOfBins = [0, 0, 0, 0, 0]
    print(wine_df['bin'][1])
    for i in range(0, wine_df.shape[0]):
        sumOfBins[wine_df['bin'][i]] = sumOfBins[wine_df['bin'][i]] + value[i]
        countOfBins[wine_df['bin'][i]] = countOfBins[wine_df['bin'][i]] + 1;

    for i  in range(0, len(sumOfBins)):
        averageOfBins[i] = sumOfBins[i]/countOfBins[i]

    for i in range(0, wine_df.shape[0]):
        value[i] = averageOfBins[wine_df['bin'][i]]


wine_df = wine_df.drop(['bin'], axis=1)
