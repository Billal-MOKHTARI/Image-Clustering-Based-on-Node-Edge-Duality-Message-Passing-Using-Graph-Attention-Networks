import pandas as pd
from sklearn.cluster import SpectralClustering, MeanShift, DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

def clustering(method, data, **kwargs):
    """
    Perform clustering on the input data using the specified method.

    Parameters:
    - method (str): The clustering method name.
    - data (pd.DataFrame): The input data.
    - **kwargs: Additional arguments specific to the clustering method.

    Returns:
    - pd.DataFrame: The input data DataFrame with an additional column indicating the cluster of each point.
    """

    if method == 'spectral':
        clustering_model = SpectralClustering(**kwargs)
        labels = clustering_model.fit_predict(data)
    elif method == 'mean_shift':
        clustering_model = MeanShift(**kwargs)
        labels = clustering_model.fit_predict(data)
    elif method == 'dbscan':
        clustering_model = DBSCAN(**kwargs)
        labels = clustering_model.fit_predict(data)
    elif method == 'kde':
        kde = KernelDensity(**kwargs)
        kde.fit(data)
        labels = kde.score_samples(data)
    elif method == 'gmm':
        gmm = GaussianMixture(**kwargs)
        labels = gmm.fit_predict(data)
    else:
        raise ValueError("Invalid clustering method specified.")

    # Add the cluster labels as a new column to the DataFrame
    data['cluster'] = labels

    return data
