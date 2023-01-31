import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..utils.conv_modules import *


class PostureTracker(nn.Module):
    def __init__(self,
                 img_shape=(100, 100),
                 n_channels_in=1,
                 n_channels_out=2,
                 init_nodes=16,
                 kernel=(3, 3),
                 padding=1,
                 pool_kernel=(2, 2)):

        super(PostureTracker, self).__init__()

        self.img_shape = img_shape
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.init_nodes = init_nodes
        self.kernel = kernel
        self.padding = padding
        self.pool_kernel = pool_kernel

        self.lstm_i = ConvLSTM(
            self.img_shape,
            self.n_channels_in,
            self.init_nodes,
            self.kernel,
            self.padding
        )
        self.conv1 = conv(
            self.init_nodes,
            self.init_nodes * 2,
            self.kernel,
            self.padding
        )
        self.conv2 = conv(
            self.init_nodes * 2,
            self.init_nodes * 4,
            self.kernel,
            self.padding
        )
        self.conv3 = conv(
            self.init_nodes * 4,
            self.init_nodes * 8,
            self.kernel,
            self.padding
        )
        self.lstm_m = ConvLSTM(
            [s // 8 for s in self.img_shape],
            self.init_nodes * 8,
            self.init_nodes * 8,
            self.kernel,
            self.padding
        )
        self.deconv1 = conv(
            self.init_nodes * 16,
            self.init_nodes * 4,
            self.kernel,
            self.padding
        )
        self.deconv2 = conv(
            self.init_nodes * 8,
            self.init_nodes * 2,
            self.kernel,
            self.padding
        )
        self.deconv3 = conv(
            self.init_nodes * 4,
            self.init_nodes,
            self.kernel,
            self.padding
        )
        self.lstm_o = ConvLSTM(
            self.img_shape,
            self.init_nodes * 2,
            self.init_nodes,
            self.kernel,
            self.padding
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(
                self.init_nodes,
                self.init_nodes,
                self.kernel,
                padding=padding,
                padding_mode='zeros'
            ),
            nn.Conv2d(
                self.init_nodes,
                self.n_channels_out,
                self.kernel,
                padding=padding,
                padding_mode='zeros'
            ),
            nn.Conv2d(
                self.n_channels_out,
                self.n_channels_out,
                self.kernel,
                padding=padding,
                padding_mode='zeros'
            ),
            # nn.BatchNorm2d(self.n_channels_out),
            # nn.Sigmoid()
        )

    def forward(self, input_tensor):
        with torch.no_grad():
            batch_size, seq_len = input_tensor.shape[:2]

        out_i = self.lstm_i(input_tensor)
        conv1 = self.conv1(out_i)
        conv2 = self.conv2(F.max_pool2d(conv1, self.pool_kernel))
        conv3 = self.conv3(F.max_pool2d(conv2, self.pool_kernel))

        pool3 = F.max_pool2d(conv3, self.pool_kernel)
        out_m = self.lstm_m(pool3.view((batch_size, seq_len, *pool3.shape[1:])))

        combined1 = torch.cat((F.interpolate(out_m, size=conv3.shape[2:]), conv3), dim=1)
        deconv1 = self.deconv1(combined1)
        combined2 = torch.cat((F.interpolate(deconv1, size=conv2.shape[2:]), conv2), dim=1)
        deconv2 = self.deconv2(combined2)
        combined3 = torch.cat((F.interpolate(deconv2, size=conv1.shape[2:]), conv1), dim=1)
        deconv3 = self.deconv3(combined3)

        combinedo = torch.cat((F.interpolate(deconv3, size=out_i.shape[2:]), out_i), dim=1)
        out_o = self.lstm_o(combinedo.view((batch_size, seq_len, *combinedo.shape[1:])))

        output_tensor = self.output_layer(out_o)

        return output_tensor.view((batch_size, seq_len, *output_tensor.shape[1:]))
