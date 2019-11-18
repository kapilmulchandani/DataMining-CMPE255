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

print(pd.cut(wine_df['proline'], 5))
print(width)
# for index, row in wine_df.iterrows():
#     if(0 < wine_df['proline'] < width):
#         wine_df