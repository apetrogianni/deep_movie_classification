"""
    This script is used to classify .mp4 movie-shots.

    Usage: 
        python3 inference.py -i <input> -m /pretrained_models/2_class/2_class_best_checkpoint.pt

        ,where <input> is the full path of an .mp4 file 
        or a folder of .mp4 files, and </pretrained_models/2_class/2_class_best_checkpoint.pt>
        is the full path of the model (in "pretrained_models" folder) to load for the prediction.

    Returns:
        a dictionary of the movie-shots and the corresponding predicted label.
"""
from operator import getitem, mod
import os
import sched
import sys
sys.path.append('..')
import torch
import pickle
import re
import argparse
import numpy as np
import torch.nn as nn
from pickle import dump, dumps
from sklearn.metrics import f1_score, accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from architectures import *
from utils.optimization import Optimization
from utils.calculate_metrics import *
from torch.utils.data import Dataset, DataLoader
from data_parsing.feature_extraction import feature_extraction
from data_parsing.dataloading import my_collate
from utils.load_dataset import LSTMDataset, load_data
from utils.data_scaling import TimeSeriesStandardScaling

def parse_arguments():
    parser = argparse.ArgumentParser(description="Wrapper"
                                                 "Predict shot class")

    parser.add_argument("-i", "--videos_path",
                        action='append', nargs='+',
                        required=True, help="Videos folder path or .mp4 file")
    parser.add_argument("-m", "--model", required=True, nargs=None,
                        help="Pretrained Model")
    parser.add_argument("-o", "--output_file", required=False, nargs=None,
                        help="Posteriors file")

    return parser.parse_args()


def predict_labels(movie, num_of_labels, model_path, model_info):
    X_test = movie
    bin_class_task = False
    if num_of_labels == 2:
        bin_class_task = True
    
    with open(model_info, "rb") as f:
        load_model = pickle.load(f)

    scaler=load_model['scaler']
    criterion = load_model['criterion']
    model = load_model['model']
    class_mapping = load_model['class_mapping']
    optimizer = load_model['optimizer']
    scheduler = load_model['scheduler']
    
    # preprocessing
    X_test, y_test, test_lengths = load_data(X_test, 0, False, \
        scaler=scaler, inference=True)
    test_dataset = LSTMDataset(X_test, y_test, test_lengths)
    test_loader = DataLoader(test_dataset, batch_size=1, \
        collate_fn=my_collate, shuffle=True)

    opt = Optimization(model=model, loss_fn=criterion, \
        optimizer=optimizer, scheduler=scheduler)

    best_model, optimizer, _, _ = \
        load_ckp(model_path, model, optimizer)

    # predict the label
    predictions = opt.evaluate(test_loader, best_model, \
        bin_class_task, inference=True)

    if num_of_labels == 2:
        # binary task
        y_pred = predictions >= 0.0
        y_pred = y_pred.float() 
    else:
        # multi-label task
        y_pred_softmax = torch.log_softmax(predictions, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        y_pred = y_pred_tags.float()
    
    return y_pred, class_mapping


if __name__ == '__main__':
    args = parse_arguments()
    videos_path = args.videos_path
    output_csv = args.output_file
    y_pred_dict = {}
    
    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]

    # load model
    trained_model = args.model
    regex = re.compile(r'\d+')
    num_of_labels = regex.search(trained_model).group(0)
    
    # get model's path
    model_path = os.path.abspath(trained_model)
    model_path = os.path.dirname(model_path)
    
    # feature_extraction(videos_path)
    videos_path = videos_path[-1]

    print("\n====================== INFERENCE ======================\n")
    pkl_files = []
    pkl_path = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(pkl_path):
            if filename.endswith("_model.pkl"):
                pkl_files.append(filename)

    for pkl_file in pkl_files:
        if num_of_labels in pkl_file:
            model_info = pkl_file

    num_of_labels = int(num_of_labels)
    # TODO: class_mapping to fix the output!
    if os.path.isdir(videos_path):
        for filename in os.listdir(videos_path):
            if filename.endswith(".mp4.npy"):
                filename = videos_path + "/" + filename
                y_pred, class_mapping = \
                    predict_labels(filename, num_of_labels, trained_model, model_info)
                y_pred_dict[filename] = int(y_pred.item())
        y_pred_dict['class_mapping'] = class_mapping

    elif os.path.isfile(videos_path):
        filename = videos_path + ".npy"
        y_pred, class_mapping = \
            predict_labels(filename, num_of_labels, trained_model, model_info)
        y_pred_dict[filename] = int(y_pred.item())
        y_pred_dict['class_mapping'] = class_mapping
    
    print(y_pred_dict)
