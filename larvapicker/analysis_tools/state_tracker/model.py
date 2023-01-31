import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTracker(nn.Module):
    def __init__(self,
                 n_channels_in=7,
                 n_channels_out=1,
                 n_chunks=1,
                 init_nodes=8,
                 dropout_rate=0.):
        super(StateTracker, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.init_nodes = init_nodes
        self.dropout_rate = dropout_rate

        self.input_layer = nn.Sequential(
            nn.Linear(n_channels_in, self.init_nodes),
            nn.ReLU(),
            nn.LayerNorm(self.init_nodes)
        )

        self.lstm_i = nn.LSTM(self.init_nodes, self.init_nodes,
                              batch_first=True, bidirectional=True)
        self.hidden_i = nn.Parameter(torch.zeros((2, n_chunks, self.init_nodes)), requires_grad=True)
        self.cell_i = nn.Parameter(torch.zeros((2, n_chunks, self.init_nodes)), requires_grad=True)

        self.linear = nn.Sequential(
            nn.PReLU(),
            nn.LayerNorm(self.init_nodes * 2),
            nn.Linear(self.init_nodes * 2, self.init_nodes * 2), nn.ReLU(),
            nn.Linear(self.init_nodes * 2, self.init_nodes * 4), nn.ReLU(),
            nn.Linear(self.init_nodes * 4, self.init_nodes * 4), nn.ReLU(),
            nn.Linear(self.init_nodes * 4, self.init_nodes * 8), nn.ReLU(),
            nn.Linear(self.init_nodes * 8, self.init_nodes * 8), nn.ReLU(),
            nn.LayerNorm(self.init_nodes * 8),
            nn.Dropout(self.dropout_rate)
        )

        self.lstm_o = nn.LSTM(self.init_nodes * 8, self.init_nodes * 4,
                              batch_first=True, bidirectional=True, num_layers=4)
        self.hidden_o = nn.Parameter(torch.zeros((8, n_chunks, self.init_nodes * 4)), requires_grad=True)
        self.cell_o = nn.Parameter(torch.zeros((8, n_chunks, self.init_nodes * 4)), requires_grad=True)

        self.output_layer = nn.Sequential(
            nn.PReLU(),
            nn.LayerNorm(self.init_nodes * 8),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.init_nodes * 8, self.init_nodes * 4), nn.ReLU(),
            nn.LayerNorm(self.init_nodes * 4),
            nn.Linear(self.init_nodes * 4, self.init_nodes), nn.ReLU(),
            nn.Linear(self.init_nodes, self.init_nodes), nn.ReLU(),
            nn.Linear(self.init_nodes, self.n_channels_out),
            # nn.Sigmoid()
        )

    def forward(self, input_tensor):
        temp = self.input_layer(input_tensor)
        temp, (h_i, c_i) = self.lstm_i(temp, (self.hidden_i, self.cell_i))
        temp = self.linear(temp)
        temp, (h_o, c_o) = self.lstm_o(temp, (self.hidden_o, self.cell_o))
        output_tensor = self.output_layer(temp)

        return output_tensor
