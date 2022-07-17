import os
import sys
sys.path.append('..')
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence as pad
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
from utils.data_scaling import TimeSeriesStandardScaling
from utils.load_dataset import LSTMDataset, load_data

print_class_map = True


def my_collate(batch):
    """
    Different padding in each batch depending on
    the longest sequence in the batch.

        Parameters
            batch: list of tuples (data, label)
    """
    # sort batch's elements in descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = pad(sequences, batch_first=True)

    # store the original length of each sequence
    # before padding (to use it for packing and unpacking)
    lengths = torch.Tensor([len(x) for x in sequences])

    # labels of the sorted batch
    sorted_labels = [item[1] for item in sorted_batch]
    labels = torch.Tensor(sorted_labels)

    return sequences_padded, labels, lengths


def data_preparation(videos_dataset, batch_size):
    # Create train, validation and test DataLoaders
    X = [x[0] for x in videos_dataset]
    y = [x[1] for x in videos_dataset]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, stratify=y)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=0.12, stratify=y_train)

    # Define Scaler
    scaler = TimeSeriesStandardScaling()

    X_train, y_train, train_lengths = \
        load_data(X_train, y_train, True, scaler=scaler)
    train_dataset = LSTMDataset(X_train, y_train, train_lengths)

    X_val, y_val, val_lengths = load_data(X_val, y_val, False, scaler=scaler)
    val_dataset = LSTMDataset(X_val, y_val, val_lengths)

    X_test, y_test, test_lengths = load_data(X_test, y_test, False, scaler=scaler)
    test_dataset = LSTMDataset(X_test, y_test, test_lengths)

    # Define a DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size,\
        collate_fn=my_collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,\
        collate_fn=my_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,\
        collate_fn=my_collate, shuffle=True)

    return train_loader, val_loader, test_loader


def create_dataset(videos_path):
    """
        Parameters:
            videos_path: a list of the full path class-folders
        Returns: 
            a list of tuples. In each tuple, the first element 
            is the video's full_path_name and the second one 
            is the corresponding y label.
    """
    global print_class_map

    class_mapping = {}
    videos_dataset = []
    label_int = -1
    for folder in videos_path:
        # folders mapping
        label = os.path.basename(folder)
        label_int += 1

        if print_class_map:
            class_mapping[label] = label_int
            # print(f" \'{label}\': {label_int}")
        for filename in os.listdir(folder):
            if filename.endswith(".mp4.npy"):
                full_path_name = folder + "/" + filename
                videos_dataset.append(tuple((full_path_name, label_int)))

    print_class_map = False
    print(class_mapping)
    return videos_dataset, class_mapping
