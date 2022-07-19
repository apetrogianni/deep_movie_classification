import sys
sys.path.append('..')
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from .calculate_metrics import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class Optimization:
    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = []
        self.val_loss = []

    def train(self, train_loader, val_loader, n_epochs, bin_class_task=True, 
              scheduler=None, optimizer=None, model=None):
        f1_max = -1.0
        counter_epoch = 0

        for epoch in range(1, n_epochs + 1):
            counter_epoch += 1
            train_losses = []
            val_losses = []
            val_predictions = []
            val_values = []

            scheduler.step(epoch)

            # enumerate mini batches
            for batch_idx, batch_info in enumerate(train_loader):
                """
                batch_idx     ---> batch id
                batch_info[0] ---> padded arrays in each batch
                batch_info[1] ---> labels (y) in each batch
                batch_info[2] ---> original length of each sequence
                """

                self.optimizer.zero_grad()

                X_train = batch_info[0]
                y_train = batch_info[1]
                X_train_original_len = batch_info[2]
                X_train_packed = \
                    pack(X_train.float(), X_train_original_len, batch_first=True)

                with torch.set_grad_enabled(True):
                    self.model.train()

                    # Compute the model output
                    out = self.model(X_train_packed, X_train_original_len)

                    # Calculate loss
                    output = out.squeeze().float()
                    if bin_class_task:
                        loss = self.loss_fn(output, y_train.float())
                    else: 
                        loss = self.loss_fn(output, \
                            y_train.type(torch.LongTensor))

                    # Computes the gradients
                    loss.backward()

                    # Updates parameters and zero gradients
                    self.optimizer.step()
                    train_losses.append(loss.item())

            train_step_loss = np.mean(train_losses)
            self.train_loss.append(train_step_loss)

            # validation process
            with torch.no_grad():
                for val_batch_idx, val_batch_info in enumerate(val_loader):
                    X_val = val_batch_info[0]
                    y_val = val_batch_info[1]
                    X_val_original_len = val_batch_info[2]
                    X_val_packed = \
                        pack(X_val.float(), X_val_original_len, batch_first=True)

                    self.model.eval()
                    
                    y_hat = self.model(X_val_packed, X_val_original_len)
                    y_hat = y_hat.squeeze().float()

                    if bin_class_task:
                        val_loss = self.loss_fn(y_hat, y_val.float())
                    else: 
                        val_loss = self.loss_fn(y_hat, y_val.type(torch.LongTensor))
                    val_losses.append(val_loss)

                    val_predictions.append(y_hat)
                    val_values.append((y_val.float()))

                if bin_class_task:
                    val_values = np.concatenate(val_values).ravel()
                    val_predictions = np.concatenate(val_predictions).ravel()
                    val_values_tensor = (torch.Tensor(val_values))
                    val_predictions_tensor = (torch.Tensor(val_predictions))

                    accuracy, f1_score_macro, cm, _ = \
                        calculate_bin_metrics(val_predictions_tensor, val_values_tensor)

                    validation_loss = np.mean(val_losses)
                    self.val_loss.append(validation_loss)
                else: 
                    val_values = np.concatenate(val_values).ravel()
                    val_predictions = np.concatenate(val_predictions)
                    val_values_tensor = (torch.Tensor(val_values))
                    val_predictions_tensor = (torch.Tensor(val_predictions))

                    accuracy, f1_score_macro, cm, _ = calculate_metrics(val_predictions_tensor, val_values_tensor)

                    validation_loss = np.mean(val_losses)
                    self.val_loss.append(validation_loss)

            # create checkpoint dictionary and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'validation_min_loss': validation_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            print(f"[{epoch}/{n_epochs}] Training loss: {train_step_loss:.4f}\t Validation loss: {validation_loss:.4f}")
            print("accuracy: {:0.2f}%,".format(accuracy * 100), "f1_score: {:0.2f}%".format(f1_score_macro * 100))

            # save checkpoint
            check_path = Path('checkpoint.pt')
            best_check_path = Path('best_checkpoint.pt')
            save_ckp(checkpoint, False, check_path, best_check_path)

            if f1_score_macro > f1_max:
                counter_epoch = 0
                print('f1_score increased({:.6f} --> {:.6f}).'.format(f1_max, f1_score_macro))
                # save checkpoint as best model
                save_ckp(checkpoint, True, check_path, best_check_path)
                f1_max = f1_score_macro

            if counter_epoch >= 15:
                break

            print("\n")

    def plot_losses(self):
        plt.title("LSTM Training and Validation Loss")
        plt.plot(self.train_loss, label="Training loss")
        plt.plot(self.val_loss, label="Validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig("LSTM_class_losses.png")
        plt.close()

    def evaluate(self, test_loader, best_model, \
        bin_class_task=True, inference=False):
        best_model.eval()
        with torch.no_grad():
            predictions = []
            values = []

            for test_batch_idx, test_batch_info in enumerate(test_loader):
                X_test = test_batch_info[0]
                y_test = test_batch_info[1] # actual values
                X_test_original_len = test_batch_info[2]
                X_test_packed = pack(X_test.float(), X_test_original_len, batch_first=True)

                y_pred = best_model(X_test_packed, X_test_original_len)
                y_pred = y_pred.squeeze(1).float()

                # retrieve numpy array
                y_pred = y_pred.detach().numpy()
                predictions.append(y_pred)

                if inference == False:
                    y_test = y_test.detach().numpy()
                    values.append(y_test)
                
        if bin_class_task:
            if inference:
                predictions = np.concatenate(predictions).ravel()
                predictions_tensor = (torch.Tensor(predictions))
                return predictions_tensor
                
            values = np.concatenate(values).ravel()
            predictions = np.concatenate(predictions).ravel()

            values_tensor = (torch.Tensor(values))
            predictions_tensor = (torch.Tensor(predictions))

            accuracy, f1_score_macro, cm, precision_recall = calculate_bin_metrics(predictions_tensor, values_tensor.float())
        else: 
            if inference:
                predictions = np.concatenate(predictions).ravel()
                predictions_tensor = (torch.Tensor(predictions))
                return predictions_tensor
            values = np.concatenate(values).ravel()
            predictions = np.concatenate(predictions)

            values_tensor = (torch.Tensor(values))
            predictions_tensor = (torch.Tensor(predictions))

            acc, f1_score_macro, cm, _ = calculate_metrics(predictions_tensor, values_tensor)

        return predictions_tensor, values_tensor, cm
