import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')


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
