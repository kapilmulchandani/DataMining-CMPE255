from sklearn.datasets import load_wine
import pandas as pd
import chart_studio.plotly as ply
import matplotlib.pylab as plt

wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(wine_df.values)

import numpy as np
mean_vec = np.mean(scaled_features, axis=0)
cov_mat = (scaled_features - mean_vec).T.dot((scaled_features - mean_vec)) / (scaled_features.shape[0]-1)

cov_mat = np.cov(scaled_features.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
list1 = var_exp
trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,14)],
    y=var_exp,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,14)],
    y=cum_var_exp,
    name='Cumulative'
)

data = [trace1, trace2]

layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)

fig = dict(data=data, layout=layout)
print(trace1['x'])
print(trace1['y'])
y_pos = np.arange(len(trace1['y']))
plt.bar(y_pos, trace1['y'], align='center')
plt.xticks(y_pos, trace1['x'])

# x, y = zip(*data) # unpack a list of pairs into two tuples
# plt.plot(x, y)
plt.show()
# ply.iplot(fig, filename='selecting-principal-components')
# ply.offline.plot(fig, filename='selecting-principal-components')
