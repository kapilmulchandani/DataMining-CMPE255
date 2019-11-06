import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sns.set(style='ticks')
data = pd.read_csv('adult_census_data.csv')
columns = data.columns.values
for i in columns:
    print(data[i].unique())
corr = data.corr()
g = sns.heatmap(corr, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5}, annot=True, fmt='.2f', cmap='coolwarm')
sns.despine()
g.figure.set_size_inches(20, 10)
plt.show()
filtered_data = data[(data['education'].isin(['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Prof-school'])) &
(data['occupation'].isin(['Exec-managerial', 'Machine-op-inspct', 'Prof-specialty' 'Sales', 'Farming-fishing', 'Tech-support', 'Armed-Forces']))]
g = sns.pairplot(filtered_data[['fnlwgt', 'education.num', 'capital.loss', 'hours.per.week', 'occupation']], hue='occupation')
sns.despine()
plt.show()
filtered_data = filtered_data.sample(n=1000, random_state=100)
g = sns.swarmplot(y = "occupation",
              x = 'hours.per.week',
              data = filtered_data,
              size = 5)

sns.despine()
g.figure.set_size_inches(12,8)
plt.show()
g = sns.boxplot(y = "occupation",
              x = 'hours.per.week',
              data = filtered_data, whis=np.inf)

g = sns.swarmplot(y = "occupation",
              x = 'hours.per.week',
              data = filtered_data,
              size = 5, color='black')
sns.despine()
g.figure.set_size_inches(12,8)
plt.show()
sns.despine()
g.figure.set_size_inches(14,10)
plt.show()