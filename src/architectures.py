import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import matplotlib
matplotlib.use('Agg')


def get_model(model, model_params):
    # choose which model to use
    models = {
        "bin_lstm": LSTMModelBinary,
        "multi_lstm": LSTMModelMulti,
    }

    return models.get(model.lower())(**model_params)


class LSTMModelBinary(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout_prob):
        super(LSTMModelBinary, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)

        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

        self.fnn = nn.Sequential(OrderedDict([
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm1d(self.hidden_size)),
            ('fc1', nn.Linear(self.hidden_size, output_size)),
        ]))

    def forward(self, X, lengths):
        packed_output, _ = self.lstm(X)
        # output shape:(batch_size,seq_length,hidden_size)
        output, _ = unpack(packed_output, batch_first=True)
        last_states = self.last_by_index(output, lengths)

        last_states = self.drop(last_states)
        output = self.fnn(last_states)

        return output

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)

        return outputs.gather(1, idx.type(torch.int64)).squeeze()


class LSTMModelMulti(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout_prob):
        super(LSTMModelMulti, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)  # , dropout=dropout_prob)

        self.fnn = nn.Sequential(OrderedDict([
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm1d(self.hidden_size)),
            ('fc1', nn.Linear(self.hidden_size, output_size)),
        ]))

        self.drop = nn.Dropout(p=dropout_prob)

    def forward(self, X, lengths):
        # Forward propagate LSTM
        packed_output, _ = self.lstm(X)
        output, _ = unpack(packed_output, batch_first=True)
        last_states = self.last_by_index(output, lengths)

        last_states = self.drop(last_states)

        output = self.fnn(last_states)

        return output

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)

        return outputs.gather(1, idx.type(torch.int64)).squeeze()
