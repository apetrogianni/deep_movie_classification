"""
Usage example:

python3 train.py -v home/3_class/Static home/3_class/Zoom home/3_class/Vertical_and_horizontal_movements

-f to run for multiple folds:
python3 train.py -v home/3_class/Static home/3_class/Zoom home/3_class/Vertical_and_horizontal_movements -f 10
"""
from copyreg import pickle
from operator import mod
import os
import sched
import sys
sys.path.append('..')
import torch
import fnmatch
import shutil
import warnings
import itertools
import argparse
import numpy as np
import torch.nn as nn
from pickle import dump, dumps
from pathlib import Path
from scipy import ndimage
from torch.nn import init
import torch.optim as optim
import sklearn.metrics as metrics
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence as pad
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
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
from data_parsing.dataloading import create_dataset, data_preparation
from data_parsing.feature_extraction import feature_extraction


def parse_arguments():
    """
    Parse arguments for training.
    """
    parser = argparse.ArgumentParser(description="Video Classifcation")

    parser.add_argument("-v", 
                        "--videos_path", 
                        required=True, 
                        action='append',
                        nargs='+', 
                        help="Movie shots folder paths" +
                        "containing the shot (mp4 and npy) files")
    parser.add_argument("-f", 
                        "--num_of_folds", 
                        required=False,
                        type=int,
                        default=1, 
                        help="Number of folds")

    # TODO: save checkpoints to another path folder

    return parser.parse_args()


def seed_all(seed):
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #random.seed(seed)
    #np.random.seed(seed)


def weight_init(m):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        #init.normal_(m.weight.data, 0.0, 0.02)
        init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    else:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


def LSTM_train(dataset, batch_size, input_size,\
    hidden_size, num_layers, output_size,
    dropout, learning_rate, weight_decay,
    choose_model, criterion, class_mapping,
    bin_class_task=True):
    
    train_loader, val_loader, test_loader = \
                data_preparation(dataset, batch_size)
            
    # LSTM params
    model_params = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'output_size': output_size,
        'dropout_prob': dropout}

    model = get_model(choose_model, model_params)    
    optimizer = optim.Adam(model.parameters(),\
        lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
        'min', verbose=True)

    model_info = {
        'criterion': criterion,
        'optimizer': optimizer,
        'batch_size': batch_size,
        'scheduler': scheduler,
        'class_mapping': class_mapping,
        'model_params': model_params,
        'model': model
    }
    
    #initialize weights for both LSTM and Sequential
    model.lstm.apply(weight_init)
    for submodule in model.fnn:
        submodule.apply(weight_init)
    
    # LSTM training
    opt = Optimization(model=model, loss_fn=criterion, \
        optimizer=optimizer, scheduler=scheduler)
    opt.train(train_loader, val_loader, n_epochs=n_epochs,\
        bin_class_task=bin_class_task, scheduler=scheduler, 
        optimizer=optimizer, model=model)

    ckp_path = "best_checkpoint.pt"
    best_model, optimizer, start_epoch, best_f1_score = \
        load_ckp(ckp_path, model, optimizer)

    predictions, values, multi_confusion_matrix = \
        opt.evaluate(test_loader, best_model, bin_class_task)

    preds.append(predictions)
    vals.append(values)

    return model_info, preds, vals


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = parse_arguments()

    videos_path = args.videos_path
    num_of_folds = args.num_of_folds
    videos_path = [item for sublist in videos_path for item in sublist]
    seed_all(42)

    preds = []
    vals = []
    model_info = {}
    num_of_shots_per_class = {}

    # extract features from .mp4 files 
    feature_extraction(videos_path)

    for folder in videos_path:
        """
        For each class-folder get the number of .npy files 
        (.npy files contain the extracted features).
        """
        class_name = os.path.basename(folder)
        np_feature_files = fnmatch.filter(os.listdir(folder), '*mp4.npy')
        num_of_shots_per_class[class_name] = len(np_feature_files)

    print("\nNumber of movie-shots per class: ", \
                num_of_shots_per_class)
    
    if len(videos_path) < 2:
        raise TypeError("You must enter at least 2 video-folders as an input!")
    if len(videos_path) == 2:
        print("\n============== BINARY CLASSIFICATION ==============")
        for i in range(0, num_of_folds):
            print(f"---- Fold {i+1}/{num_of_folds} ----")
            # create LSTM dataset & DataLoaders
            dataset, class_mapping = create_dataset(videos_path)

            n_epochs = 100
            input_size = 43
            num_layers = 2
            batch_size = 70
            hidden_size = 100
            dropout = 0.5
            lr = 1e-3
            weight_decay = 1e-5
            output_size = 1

            criterion = nn.BCEWithLogitsLoss()

            model_info, preds, vals = \
                LSTM_train(dataset=dataset,batch_size=batch_size,
                input_size=input_size, hidden_size=hidden_size, 
                num_layers=num_layers, output_size=output_size, 
                dropout=dropout, learning_rate=lr, 
                weight_decay=weight_decay, choose_model='bin_lstm',
                criterion=criterion, class_mapping=class_mapping, 
                bin_class_task=True)  

            # Save Loss Function, optimizer, scheduler,
            # batch_size, model_params and the model in a .pkl file
            with open('binary_best_model.pkl', 'wb') as f:
                dump(model_info, f)


        preds = torch.Tensor(np.concatenate(preds).ravel())
        vals = np.concatenate(vals).ravel()
        class_labels = list(set(vals))
        vals = torch.Tensor(vals)

        np.save("binary_LSTM_y_test.npy", vals)
        np.save("binary_LSTM_y_pred.npy", preds)

        accuracy, f1_score_macro, cm, class_labels, precision_recall = \
            calculate_bin_aggregated_metrics(preds, vals.float(), class_labels)

        print(f"{num_of_folds}-fold Classification Report:\n"
            "accuracy: {:0.2f}%,".format(accuracy * 100),
            "precision: {:0.2f}%,".format(precision_recall[0] * 100),
            "recall: {:0.2f}%,".format(precision_recall[1] * 100),
            "f1_score (macro): {:0.2f}%".format(f1_score_macro * 100))
        print("\nConfusion matrix\n", cm)

        np.set_printoptions(precision=2)
        plot_bin_confusion_matrix('LSTM', cm, classes=class_labels)
        os.remove("checkpoint.pt")
        print("==============-----------------------==============")
    else: 
        print("\n=============== MULTI-LABEL CLASSIFICATION ===============")
        num_of_shots_per_class_list = list(num_of_shots_per_class.values()) 
        major_class = max(num_of_shots_per_class_list)
        weights = []
        # minor_class = min(num_of_shots_per_class_list)
        # for class_folder_shots in num_of_shots_per_class_list:
        #     weight_class = minor_class / class_folder_shots
        #     weights.append(weight_class)

        for class_folder_shots in num_of_shots_per_class_list:
            weight_class = major_class / class_folder_shots
            weights.append(weight_class)

        weights = torch.FloatTensor(weights)

        for i in range(0, num_of_folds):
            print(f"---- Fold {i+1}/{num_of_folds} ----")
            # create LSTM dataset & DataLoaders
            dataset, class_mapping = create_dataset(videos_path)

            # Select parameters for each experiment
            if len(videos_path) == 3:
                n_epochs = 100
                input_size = 43 
                num_layers = 1
                batch_size = 32
                hidden_size = 64
                dropout = 0.1
                lr = 1e-2
                weight_decay = 1e-8
                output_size = len(videos_path)
                multi_model= '3_class_lstm'
            elif len(videos_path) == 4:
                n_epochs = 100
                input_size = 43 
                num_layers = 1
                batch_size = 32
                hidden_size = 40
                dropout = 0.0
                lr = 1e-2
                weight_decay = 0.0
                output_size = len(videos_path)
                multi_model= '4_class_lstm'
            elif len(videos_path) == 10:
                n_epochs = 100
                input_size = 43 
                num_layers = 1
                batch_size = 13
                hidden_size = 35
                dropout = 0.2
                lr = 1e-3
                weight_decay = 1e-8
                output_size = len(videos_path)
                multi_model= '10_class_lstm'

            criterion = nn.CrossEntropyLoss(weight=weights)

            model_info, preds, vals = \
                LSTM_train(dataset=dataset,batch_size=batch_size,\
                input_size=input_size, hidden_size=hidden_size, 
                num_layers=num_layers, output_size=output_size, 
                dropout=dropout, learning_rate=lr, weight_decay=weight_decay, 
                choose_model=multi_model, criterion=criterion, 
                class_mapping=class_mapping, bin_class_task=False)  
            
            # Save Loss Function, optimizer, scheduler,
            # batch_size, model_params and the model in a .pkl file
            with open(str(len(videos_path))+'_class_best_model.pkl', 'wb') as f:
                dump(model_info, f)
            
        vals = np.concatenate(vals).ravel()
        preds = torch.Tensor(np.concatenate(preds))
        class_labels = list(set(vals))
        vals = torch.Tensor(vals)

        np.save(str(len(videos_path)) + "_class_LSTM_" + "_y_test.npy", vals)
        np.save(str(len(videos_path)) + "_class_LSTM_" + "_y_pred.npy", preds)

        accuracy, f1_score_macro, cm, class_labels = \
            calculate_aggregated_metrics(preds, vals, class_labels)

        print(f"\n{num_of_folds}-fold Classification Report:\n"
            "accuracy: {:0.2f}%,".format(accuracy * 100),
            "f1_score (macro): {:0.2f}%".format(f1_score_macro * 100))
        print("\nConfusion matrix\n", cm)

        np.set_printoptions(precision=2)
        plot_confusion_matrix('LSTM', cm, videos_path=videos_path, classes=class_labels)
        os.remove("checkpoint.pt")
        print("=======----------------------------=======")
