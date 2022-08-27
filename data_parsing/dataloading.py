import os
import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence as pad
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
from utils.data_scaling import TimeSeriesStandardScaling, TimeSeriesPCA
from utils.load_dataset import (LSTMDataset, load_data)

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


def data_preparation(videos_dataset, batch_size, pretrained=None):
    # Define Scaler
    scaler = TimeSeriesStandardScaling()

    if pretrained is not None:
        pca = TimeSeriesPCA(n_components=1024)
    else:
        pca=None

    # Create train, validation and test DataLoaders
    X = [x[0] for x in videos_dataset]
    y = [x[1] for x in videos_dataset]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, stratify=y)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=0.12, stratify=y_train)

    X_train, y_train, train_lengths, pca = \
        load_data(X_train, y_train, True, scaler=scaler, pca=pca, pretrained=pretrained)
    train_dataset = LSTMDataset(X_train, y_train, train_lengths)

    num_of_features = X_train[0].size(1)
    if pretrained is not None:
        print("Number of features, after applying PCA: ", num_of_features)
    else: 
        num_of_features = 43

    X_val, y_val, val_lengths, _ = \
        load_data(X_val, y_val, False, scaler=scaler, pca=pca, pretrained=pretrained)
    val_dataset = LSTMDataset(X_val, y_val, val_lengths)

    X_test, y_test, test_lengths, _ = \
        load_data(X_test, y_test, False, scaler=scaler, pca=pca, pretrained=pretrained)
    test_dataset = LSTMDataset(X_test, y_test, test_lengths)

    # Define a DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=my_collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=my_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             collate_fn=my_collate, shuffle=True)

    return train_loader, val_loader, test_loader, scaler, pca, num_of_features


def create_dataset(videos_path, pretrained = None):
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
            if pretrained == 'VGG':
                if filename.endswith("VGG16.npy"):
                    full_path_name = folder + "/" + filename
            elif pretrained == 'googlenet':
                if filename.endswith("_googlenet.npy"):
                    full_path_name = folder + "/" + filename
            elif pretrained == 'densenet201':
                if filename.endswith("_densenet201.npy"):
                    full_path_name = folder + "/" + filename
            elif pretrained == 'vgg19_bn':
                if filename.endswith("_vgg19_bn.npy"):
                    full_path_name = folder + "/" + filename
            elif pretrained == 'inceptionv3':
                if filename.endswith("_inceptionv3.npy"):
                    full_path_name = folder + "/" + filename
            elif pretrained == 'shufflenetv2':
                if filename.endswith("_shufflenetv2.npy"):
                    full_path_name = folder + "/" + filename
            elif filename.endswith(".mp4.npy"):
                # hand-crafted features (multimodal_movie_analysis library)
                full_path_name = folder + "/" + filename
            
            videos_dataset.append(tuple((full_path_name, label_int)))

    print_class_map = False
    print(class_mapping)
    return videos_dataset, class_mapping
