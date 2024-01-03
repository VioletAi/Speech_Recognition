import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_rate):
        super().__init__()

        # LSTM layer with dropout
        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=False)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Projection layer
        self.proj = nn.Linear(hidden_dims, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        # Applying dropout
        hidden = self.dropout(hidden)
        output = self.proj(hidden)
        return output
