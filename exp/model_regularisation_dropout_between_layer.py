import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_rate):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, dropout=dropout_rate, bidirectional=True)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output
