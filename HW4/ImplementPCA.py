from sklearn.datasets import load_wine
import pandas as pd

wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(wine_df.values)

import numpy as np
mean_vec = np.mean(scaled_features, axis=0)
cov_mat = (scaled_features - mean_vec).T.dot((scaled_features - mean_vec)) / (scaled_features.shape[0]-1)

cov_mat = np.cov(scaled_features.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
