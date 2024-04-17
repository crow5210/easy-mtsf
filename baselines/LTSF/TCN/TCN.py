from torch import nn
import torch
import math
from utils.__init__ import TSDecoder
import torch.nn.functional as F
class ResidualBlock(nn.Module):

    def __init__(self, num_filters: int, kernel_size: int, dilation_base: int, dropout: float, weight_norm: bool, nr_blocks_below: int, num_layers: int, input_size: int, target_size: int):
        super(ResidualBlock, self).__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.activation = nn.ReLU()
        self.dropout_fn = nn.Dropout(p=dropout) # dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_size = input_size if nr_blocks_below == 0 else num_filters
        target_size = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, dilation=(dilation_base ** nr_blocks_below))
        self.conv2 = nn.Conv1d(num_filters, target_size, kernel_size, dilation=(dilation_base ** nr_blocks_below))
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(self.conv1), nn.utils.weight_norm(self.conv2)
        self.conv3 = nn.Conv1d(input_size, target_size, 1) if input_size != target_size else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.conv3 is not None: self.conv3.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base ** self.nr_blocks_below) * (self.kernel_size - 1)
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(self.activation(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = self.activation(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x += residual

        return x

class TCN(nn.Module):
    '''
    Temporal Convolutional Network
    ------------------------------
    refer: 
    1. _ResidualBlock, _TCNModule ***
    in darts-master\darts\models\forecasting\tcn_model.py
    2. TemporalBlock, TCN ****
    in tsai-main\tsai\models\TCN.py
    '''
    def __init__(self, n_in: int = 8, n_out: int = 1, input_size: int = 3, target_size: int = 3, kernel_size: int = 7, num_filters: int = 32, num_layers = 5, dilation_base: int = 2, weight_norm: bool = True, dropout: float = 0.):
        super(TCN, self).__init__() #None #False

        # Defining parameters
        self.input_size = input_size
        self.n_in = n_in
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.n_out = n_out
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = dropout # nn.Dropout(p=dropout)

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(math.log((n_in - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1, dilation_base))
            print("[TCN] Number of layers chosen: " + str(num_layers) + '\r',end='')
        elif num_layers is None:
            num_layers = math.ceil((n_in - 1) / (kernel_size - 1) / 2)
            print("[TCN] Number of layers chosen: " + str(num_layers) + '\r',end='')
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = ResidualBlock(num_filters, kernel_size, dilation_base, self.dropout, weight_norm, i, num_layers, self.input_size, target_size)
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)
        # self.fc = nn.Linear(n_in, n_out)
        ### refer TS2Vec
        self.tsdecoder = TSDecoder(n_in=n_in, n_out=n_out, input_size=input_size, target_size=target_size)#, dec_mode='fusing2'

    def forward_tcn(self, x):
        # data is of size (batch_size, n_in, input_size)
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        for i in range(self.num_layers):
            x = self.res_blocks[i](x) #self.res_blocks_list:

        x = x.transpose(1, 2)
        x = x.view(batch_size, self.n_in, self.target_size)

        output = self.tsdecoder(x)

        return output #self.fc(x.permute(0,2,1)).permute(0,2,1)
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        return self.forward_tcn(history_data[...,0]).unsqueeze(-1)