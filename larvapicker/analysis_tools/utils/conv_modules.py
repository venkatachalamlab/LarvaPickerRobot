import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self,
                 n_channels_in,
                 n_channels_out,
                 kernel=(3, 3),
                 padding=1):
        super(ConvLSTMCell, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.kernel = kernel
        self.padding = padding

        self.conv = nn.Conv2d(
            self.n_channels_in + self.n_channels_out,
            4 * self.n_channels_out,
            self.kernel,
            padding=self.padding,
            padding_mode='zeros'
        )

    def forward(self, input_tensor, hidden_tensor, cell_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        combined = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.n_channels_out, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        cell_tensor = f * cell_tensor + i * g
        hidden_tensor = o * torch.tanh(cell_tensor)
        return hidden_tensor, cell_tensor

class ConvLSTM(nn.Module):
    def __init__(self,
                 shape_in,
                 n_channels_in,
                 n_channels_out,
                 kernel=(3, 3),
                 padding=1):
        super(ConvLSTM, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out

        self.hidden_init = nn.Parameter(torch.zeros((1, n_channels_out, *shape_in)), requires_grad=True)
        self.cell_init = nn.Parameter(torch.zeros((1, n_channels_out, *shape_in)), requires_grad=True)

        self.cell = ConvLSTMCell(
            self.n_channels_in,
            self.n_channels_out,
            kernel,
            padding
        )

        # self.actv = nn.Sequential(
        #     nn.BatchNorm2d(self.n_channels_out),
        #     nn.ReLU()
        # )

    def forward(self, input_tensor):
        h_t = self.hidden_init.expand((input_tensor.shape[0], -1, -1, -1))
        c_t = self.cell_init.expand((input_tensor.shape[0], -1, -1, -1))
        output_tensor = []
        for t in range(input_tensor.shape[1]):
            h_t, c_t = self.cell(input_tensor[:, t, ...], h_t, c_t)
            output_tensor.append(h_t)
        return torch.cat(output_tensor, dim=0)

def conv(n_channels_in,
         n_channels_out,
         kernel=(3, 3),
         padding=1):

    return nn.Sequential(
        nn.Conv2d(n_channels_in,
                  n_channels_out,
                  kernel,
                  padding=padding,
                  padding_mode='zeros'),
        nn.BatchNorm2d(n_channels_out),
        nn.ReLU(),
        nn.Conv2d(n_channels_out,
                  n_channels_out,
                  kernel,
                  padding=padding,
                  padding_mode='zeros'),
        nn.BatchNorm2d(n_channels_out),
        nn.ReLU()
    )
