import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence as pad
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

print_class_map = True


class LSTMDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = X
        self.y = y
        self.lengths = lengths

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.lengths[index]


def load_data(X_in, y, check_train, scaler, inference=False):
    # for train/val/test sets

    if inference:
        X_to_tensor = np.load(X_in)
        
        x_len = []
        X = []
        label = []
        
        # keep only specific features
        # (remove histograms)
        X_to_tensor = X_to_tensor[:, 45:89]
        X_to_tensor = np.array([ndimage.median_filter(s, 4)
                                for s in X_to_tensor.T]).T

        label = [-1]
        x_len.append(X_to_tensor.shape[0])
        X.append(torch.Tensor(X_to_tensor))

        X_scaled = scaler.transform(X)

        return X_scaled, label, x_len
    else:
        split_dataset = []
        for i, j in zip(X_in, y):
            split_dataset.append(tuple((i, j)))

        x_len = []
        X = []
        labels = []
        
        for index, data in enumerate(split_dataset):
            # print(split_dataset)
            """
            data[0] corresponds to--->.npy movie-shot names
            data[1] corresponds to--->y labels
            """
            X_to_tensor = np.load(data[0])

            # keep only specific features
            # (remove histograms)
            X_to_tensor = X_to_tensor[:, 45:89]
            X_to_tensor = np.array([ndimage.median_filter(s, 4)
                                    for s in X_to_tensor.T]).T

            y = data[1]
            labels.append(y)
            x_len.append(X_to_tensor.shape[0])
            X.append(torch.Tensor(X_to_tensor))

        # data normalization
        if check_train:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

    return X_scaled, labels, x_len