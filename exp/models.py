import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        print(feat.shape)
        hidden, _ = self.lstm(feat)
        print(hidden.shape)
        output = self.proj(hidden)
        
        print(output.shape)
        print("\n")
        return output
