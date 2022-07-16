"""
Usage example:
python3 train.py -v <class_folder_name> <class_folder_name> ... <class_folder_name>

i.e.
python3 train.py -v ../3_class/Static ../3_class/Zoom ../3_class/Vertical_and_horizontal_movements

-f to run for multiple folds

python3 train.py -v /media/antonia/Seagate/datasets/Hand_Crafted_dataset/2_class/Non_Static /media/antonia/Seagate/datasets/Hand_Crafted_dataset/2_class/Static
"""
from operator import mod
import os
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


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = parse_arguments()

    videos_path = args.videos_path
    num_of_folds = args.num_of_folds
    videos_path = [item for sublist in videos_path for item in sublist]
    seed_all(42)

    preds = []
    vals = []

    num_of_shots_per_class = {}

    feature_extraction(videos_path)

    # for folder in videos_path:
    #     """
    #     For each class-folder get the numbrt of npy files 
    #     (.npy files contain the extracted features).
    #     """
    #     class_name = os.path.basename(folder)
    #     np_feature_files = fnmatch.filter(os.listdir(folder), '*mp4.npy')
    #     num_of_shots_per_class[class_name] = len(np_feature_files)
    
    # if len(videos_path) < 2:
    #     raise TypeError("You must enter 2 video-folders at least!")
    # if len(videos_path) == 2:
    #     print("\n============== BINARY CLASSIFICATION ==============")
    #     for i in range(0, num_of_folds):
    #         n_epochs = 1
    #         output_size = 1
    #         input_size = 43  # num of features
    #         hidden_size = 100
    #         num_layers = 2
    #         batch_size = 70
    #         dropout = 0.4
    #         learning_rate = 1e-3
    #         weight_decay = 1e-5
            
    #         print("\nNumber of movie-shots per class: ", num_of_shots_per_class)

    #         # create LSTM dataset & DataLoaders
    #         dataset = create_dataset(videos_path)
    #         train_loader, val_loader, test_loader = \
    #             data_preparation(dataset, batch_size=batch_size)
            
    #         print(f"---- Fold {i+1}/{num_of_folds} ----")
            
    #         # LSTM params 
    #         model_params = {'input_size': input_size,
    #                         'hidden_size': hidden_size,
    #                         'num_layers': num_layers,
    #                         'output_size': output_size,
    #                         'dropout_prob': dropout}
    #         model = get_model('bin_lstm', model_params)

    #         criterion = nn.BCEWithLogitsLoss()
    #         optimizer = optim.Adam(model.parameters(),\
    #             lr=learning_rate, weight_decay=weight_decay)
    #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
    #             'min', verbose=True)

    #         # TODO: save them DO NOT print them !
    #         # print(f"{model_params}" +\
    #         #     f"\nbatch_size: {batch_size}" +\
    #         #     f"\nLoss function: {criterion}" +\
    #         #     f"\nOptimizer: {optimizer}" +\
    #         #     f"\n{model}")

    #         # initialize weights for both LSTM and Sequential
    #         model.lstm.apply(weight_init)
    #         for submodule in model.fnn:
    #             submodule.apply(weight_init)

    #         # LSTM training
    #         opt = Optimization(model=model, loss_fn=criterion,\
    #             optimizer=optimizer, scheduler=scheduler)
    #         opt.train(train_loader, val_loader, n_epochs=n_epochs,bin_class_task=True,
    #         scheduler=scheduler, optimizer=optimizer, model=model)
    #         # opt.plot_losses()

    #         # Evaluation
    #         ckp_path = "best_checkpoint.pt"
    #         best_model, optimizer, start_epoch, best_f1_score = \
    #             load_ckp(ckp_path, model, optimizer)

    #         predictions, values, binary_confusion_matrix = \
    #             opt.evaluate(test_loader, best_model,bin_class_task=True)

    #         preds.append(predictions)
    #         vals.append(values)

    #     preds = torch.Tensor(np.concatenate(preds).ravel())
    #     vals = np.concatenate(vals).ravel()
    #     class_labels = list(set(vals))
    #     vals = torch.Tensor(vals)

    #     np.save("LSTM_binary_y_test.npy", vals)
    #     np.save("LSTM_binary_y_pred.npy", preds)

    #     accuracy, f1_score_macro, cm, class_labels, precision_recall = \
    #         calculate_bin_aggregated_metrics(preds, vals.float(), class_labels)

    #     print(f"{num_of_folds}-fold Classification Report:\n"
    #         "accuracy: {:0.2f}%,".format(accuracy * 100),
    #         "precision: {:0.2f}%,".format(precision_recall[0] * 100),
    #         "recall: {:0.2f}%,".format(precision_recall[1] * 100),
    #         "f1_score (macro): {:0.2f}%".format(f1_score_macro * 100))
    #     print("\nConfusion matrix\n", cm)

    #     np.set_printoptions(precision=2)
    #     plot_bin_confusion_matrix('LSTM', cm, classes=class_labels)
    #     print("==============-----------------------==============")
    # else: 
    #     print("\n======= MULTI-LABEL CLASSIFICATION =======")

    #     print("\nNumber of movie-shots per class: ", num_of_shots_per_class)

    #     minor_class = min(num_of_shots_per_class)
    #     major_class = max(num_of_shots_per_class)

    #     weights = []
    #     # for class_folder_shots in num_of_shots_per_class:
    #     #     weight_class = minor_class / class_folder_shots
    #     #     weights.append(weight_class)

    #     for class_folder_shots in num_of_shots_per_class:
    #         weight_class = major_class / class_folder_shots
    #         weights.append(weight_class)

    #     weights = torch.FloatTensor(weights)

    #     for i in range(0, num_of_folds):
    #         n_epochs = 2
    #         input_size = 43
    #         num_layers = 1
    #         batch_size = 32
    #         hidden_size = 64
    #         dropout = 0.1
    #         learning_rate = 1e-2
    #         weight_decay = 1e-8
    #         output_size = len(videos_path)

    #         # create LSTM dataset & DataLoaders
    #         dataset = create_dataset(videos_path)
    #         train_loader, val_loader, test_loader = data_preparation(
    #             dataset, batch_size=batch_size)

    #         print(f"---- Fold {i+1}/{num_of_folds} ----")
    #         # LSTM params
    #         model_params = {'input_size': input_size,
    #                         'hidden_size': hidden_size,
    #                         'num_layers': num_layers,
    #                         'output_size': output_size,
    #                         'dropout_prob': dropout}
            
    #         model = get_model('multi_lstm', model_params)

    #         # TODO: save them DO NOT print them
    #         # print(f"{model_params}" +\
    #         #     f"\nbatch_size: {batch_size}" +\
    #         #     f"\nLoss function: {criterion}" +\
    #         #     f"\nOptimizer: {optimizer}" +\
    #         #     f"\n{model}")
    #         criterion = nn.CrossEntropyLoss(weight=weights)
    #         optimizer = optim.Adam(model.parameters(),
    #                             lr=learning_rate, weight_decay=weight_decay)
    #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
            
    #         #initialize weights for both LSTM and Sequential
    #         model.lstm.apply(weight_init)
    #         for submodule in model.fnn:
    #             submodule.apply(weight_init)
            
    #         opt = Optimization(model=model, loss_fn=criterion, optimizer=optimizer, scheduler=scheduler)

    #         opt.train(train_loader, val_loader, n_epochs=n_epochs, bin_class_task=False)
    #         # opt.plot_losses()

    #         ckp_path = "best_checkpoint.pt"
    #         best_model, optimizer, start_epoch, best_f1_score = \
    #             load_ckp(ckp_path, model, optimizer)

    #         predictions, values, multi_confusion_matrix = \
    #             opt.evaluate(test_loader, best_model,bin_class_task=False)

    #         preds.append(predictions)
    #         vals.append(values)


    #     vals = np.concatenate(vals).ravel()
    #     preds = torch.Tensor(np.concatenate(preds))
    #     class_labels = list(set(vals))
    #     vals = torch.Tensor(vals)

    #     np.save("LSTM_" + str(len(videos_path)) + "_class_y_test.npy", vals)
    #     np.save("LSTM_" + str(len(videos_path)) + "_class_y_pred.npy", preds)

    #     accuracy, f1_score_macro, cm, class_labels = \
    #         calculate_aggregated_metrics(preds, vals, class_labels)

    #     print(f"{num_of_folds}-fold Classification Report:\n"
    #         "accuracy: {:0.2f}%,".format(accuracy * 100),
    #         # "precision: {:0.2f}%,".format(precision_recall[0] * 100),
    #         # "recall: {:0.2f}%,".format(precision_recall[1] * 100),
    #         "f1_score (macro): {:0.2f}%".format(f1_score_macro * 100))
    #     print("\nConfusion matrix\n", cm)

    #     np.set_printoptions(precision=2)
    #     plot_confusion_matrix('LSTM', cm, videos_path=videos_path, classes=class_labels)
    #     print("=======----------------------------=======")
