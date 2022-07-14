import torch
import shutil
import itertools
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig("ROCurve_binary.png")
    plt.close()


def plot_precision_recall_curve(precision, recall, y_test):
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='LSTM')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    #pyplot.show()
    plt.savefig("Precision_Recall_Curve_binary.png")
    plt.close()


def calculate_bin_metrics(predicted_values, actual_values, threshold=0.0):

    y_pred = predicted_values >= threshold
    y = actual_values
    y_pred = y_pred.float()

    cm = confusion_matrix(y, y_pred)
    f1_score_macro = f1_score(y, y_pred, average='macro')
    acc = accuracy_score(y, y_pred)
    precision_recall_fscore = precision_recall_fscore_support(y, y_pred, average='macro')

    return acc, f1_score_macro, cm, precision_recall_fscore
    

def calculate_bin_aggregated_metrics(predicted_values, actual_values, class_labels, threshold=0.0):

    y_pred = predicted_values >= threshold
    y = actual_values

    y_pred = y_pred.float()

    cm = confusion_matrix(y, y_pred, labels=class_labels)
    f1_score_macro = f1_score(y, y_pred, average='macro')
    acc = accuracy_score(y, y_pred)
    precision_recall_fscore = precision_recall_fscore_support(y, y_pred, average='macro')

    # ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(actual_values, predicted_values)
    precision, recall, thresholds = metrics.precision_recall_curve(actual_values, predicted_values)
    roc_auc = metrics.auc(fpr, tpr)

    plot_roc_curve(fpr, tpr, roc_auc)
    plot_precision_recall_curve(precision, recall, actual_values)

    return acc, f1_score_macro, cm, class_labels, precision_recall_fscore


def plot_bin_confusion_matrix(name, cm, classes):
    """
    Plot confusion matrix
    :name: name of classifier
    :cm: estimates of confusion matrix
    :classes: all the classes
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(str(name) + '_binary_cm' + '.eps', format='eps')


def calculate_metrics(y_pred, y_test, id=0):

    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    y_pred = y_pred_tags.float()

    cm = confusion_matrix(y_test, y_pred)
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    precision_recall_fscore = precision_recall_fscore_support(y_test, y_pred, average='macro')

    return acc, f1_score_macro, cm, precision_recall_fscore

def calculate_aggregated_metrics(y_pred, y_test, class_labels):

    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    y_pred = y_pred_tags.float()

    print(class_labels)

    conf_mat = confusion_matrix(y_test, y_pred)
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    #precision_recall_fscore = precision_recall_fscore_support(y_test, y_pred, average='macro')

    return acc, f1_score_macro, conf_mat, class_labels #, precision_recall_fscore


def plot_confusion_matrix(name, cm, videos_path, classes):
    """
    Plot confusion matrix
    :name: name of classifier
    :cm: estimates of confusion matrix
    :classes: all the classes
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(str(len(videos_path)) + '_shot_classifier_conf_mat_' + str(name) + '.eps', format='eps')




def save_ckp(checkpoint, is_best_val, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    """

    # save checkpoint data
    torch.save(checkpoint, checkpoint_path)

    # if it is the best model
    if is_best_val:
        best_check_path = best_model_path

        # copy that checkpoint file to best path
        shutil.copyfile(checkpoint_path, best_check_path)


def load_ckp(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    validation_min_loss = checkpoint['validation_min_loss']

    return model, optimizer, checkpoint['epoch'], validation_min_loss.item()