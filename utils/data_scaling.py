"""
Create both a StandardScaler and a MinMaxScaler
to normalize the data column-wise (for each feature).
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')


def zip_time(X):
    X_extended = []
    lengths = []
    for x in X:
        lengths.append(x.shape[0])
        for row in x.cpu().detach().numpy():
            X_extended.append(row)
    return X_extended, lengths


def reconstruct_time_dataset(X, lengths):
    X_reconstructed = []
    X_rest = X
    for length in lengths:
        X_current = X_rest[:length, :]
        X_rest = X_rest[length:, :]
        X_current = torch.Tensor(X_current)
        X_reconstructed.append(X_current)

    return X_reconstructed


class TimeSeriesPCA():
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit(self, X):
        self.X = X
        X_extended, _ = zip_time(X)
        self.pca.fit(X_extended)
        print("variance: ", self.pca.explained_variance_ratio_.sum())
        print("n_componenets: ", self.pca.components_.shape[0])

    def transform(self, X):
        X_extended, lengths = zip_time(X)
        X_extended_dec = self.pca.transform(X_extended)

        X_dec = reconstruct_time_dataset(X_extended_dec, lengths)
        return X_dec

    def fit_transform(self, X):
        self.fit(X)
        X_dec = self.transform(self.X)

        return X_dec


class TimeSeriesStandardScaling():
    def __init__(self):
        pass

    def fit(self, X):
        self.X = X
        k = 0
        mean_std_tuples = []
        while k < self.X[0].shape[1]:
            temp = np.concatenate([i[:, k] for i in self.X])
            mean_std_tuples.append(
                tuple((np.mean(temp), np.std(temp))))
            k += 1
        self.scale_params = mean_std_tuples

    def transform(self, X):
        X_scaled = []
        tmp = list(zip(*self.scale_params))
        mu = torch.Tensor(list(tmp[0]))
        std = torch.Tensor(list(tmp[1]))

        for inst in X:
            v = (inst - mu) / std
            X_scaled.append(v)
        self.X_scaled = X_scaled

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        self.transform(self.X)
        return self.X_scaled


class TimeSeriesMinMaxScaling():
    def __init__(self):
        pass

    def fit(self, X):
        self.X = X
        k = 0
        min_max_tuples = []
        while k < self.X[0].shape[1]:
            temp = np.concatenate([i[:, k] for i in self.X])
            min_max_tuples.append(
                tuple((np.min(temp), np.max(temp))))
            k += 1

        self.scale_params = min_max_tuples

    def transform(self, X):
        X_scaled = []
        tmp = list(zip(*self.scale_params))
        min_feature = torch.Tensor(list(tmp[0]))
        max_feature = torch.Tensor(list(tmp[1]))

        for inst in X:
            v = (inst - min_feature) / (max_feature - min_feature)
            X_scaled.append(v)
        self.X_scaled = X_scaled

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        self.transform(self.X)
        return self.X_scaled
