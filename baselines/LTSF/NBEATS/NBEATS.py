import torch
from torch import nn
from typing import NewType, Union, List, Optional, Tuple
from enum import Enum
import numpy as np

class _GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3


GTypes = NewType('GTypes', _GType)


class TrendGenerator(nn.Module):

    def __init__(self,
                 expansion_coefficient_dim,
                 n_out):
        super(TrendGenerator, self).__init__()

        # basis is of size (expansion_coefficient_dim, n_out)
        basis = torch.stack([(torch.arange(n_out) / n_out)**i for i in range(expansion_coefficient_dim)], 
                            dim=1).T

        self.basis = nn.Parameter(basis, requires_grad=False)


    def forward(self, x):
        return torch.matmul(x, self.basis)


class SeasonalityGenerator(nn.Module):

    def __init__(self,
                 n_out):
        super(SeasonalityGenerator, self).__init__()
        half_minus_one = int(n_out / 2 - 1)
        cos_vectors = [torch.cos(torch.arange(n_out) * 2 * np.pi * i) for i in range(1, half_minus_one + 1)]
        sin_vectors = [torch.sin(torch.arange(n_out) * 2 * np.pi * i) for i in range(1, half_minus_one + 1)]
        
        # basis is of size (2 * int(n_out / 2 - 1) + 1, n_out)
        basis = torch.stack([torch.ones(n_out)] + cos_vectors + sin_vectors, dim=1).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)

class NbeatsBlock(nn.Module):

    def __init__(self,
                 num_layers: int,
                 layer_width: int,
                 expansion_coefficient_dim: int,
                 n_in: int,
                 n_out: int,
                 g_type: GTypes):
        super(NbeatsBlock, self).__init__()

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.n_out = n_out
        self.g_type = g_type
        self.relu = nn.ReLU()

        # fully connected stack before fork
        self.linear_layer_stack_list = [nn.Linear(n_in, layer_width)]
        self.linear_layer_stack_list += [nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.fc_stack = nn.ModuleList(self.linear_layer_stack_list)

        # Fully connected layer producing forecast/backcast expansion coeffcients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood.
        if g_type == _GType.SEASONALITY:
            self.backcast_linear_layer = nn.Linear(layer_width, 2 * int(n_in / 2 - 1) + 1)
            self.forecast_linear_layer = nn.Linear(layer_width, (2 * int(n_out / 2 - 1) + 1))
        else:
            self.backcast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)
            self.forecast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)

        # waveform generator functions
        if g_type == _GType.GENERIC:
            self.backcast_g = nn.Linear(expansion_coefficient_dim, n_in)
            self.forecast_g = nn.Linear(expansion_coefficient_dim, n_out)
        elif g_type == _GType.TREND:
            self.backcast_g = TrendGenerator(expansion_coefficient_dim, n_in)
            self.forecast_g = TrendGenerator(expansion_coefficient_dim, n_out)
        elif g_type == _GType.SEASONALITY:
            self.backcast_g = SeasonalityGenerator(n_in)
            self.forecast_g = SeasonalityGenerator(n_out)
        else:
            print(ValueError("g_type not supported"))

    def forward(self, x):
        batch_size = x.shape[0]

        # fully connected layer stack
        for layer in self.linear_layer_stack_list:
            x = self.relu(layer(x))

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # set the expansion coefs in last dimension for the forecasts
        theta_forecast = theta_forecast.view(batch_size, -1)

        # waveform generator applications (project the expansion coefs onto basis vectors)
        x_hat = self.backcast_g(theta_backcast)
        y_hat = self.forecast_g(theta_forecast)

        # Set the distribution parameters as the last dimension
        y_hat = y_hat.reshape(x.shape[0], self.n_out)

        return x_hat, y_hat


class NbeatsStack(nn.Module):

    def __init__(self,
                 num_blocks: int,
                 num_layers: int,
                 layer_width: int,
                 expansion_coefficient_dim: int,
                 n_in: int,
                 n_out: int,
                 g_type: GTypes,
                 ):
        super(NbeatsStack, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

        if g_type == _GType.GENERIC:
            self.blocks_list = [
                NbeatsBlock(num_layers, layer_width, 
                       expansion_coefficient_dim, n_in, 
                       n_out, g_type)
                for _ in range(num_blocks)
            ]
        else:
            # same block instance is used for weight sharing
            interpretable_block = NbeatsBlock(num_layers, layer_width,
                                         expansion_coefficient_dim, n_in, 
                                         n_out, g_type)
            self.blocks_list = [interpretable_block] * num_blocks

        self.blocks = nn.ModuleList(self.blocks_list)

    def forward(self, x):
        # One forecast vector per parameter in the distribution
        stack_forecast = torch.zeros(x.shape[0], 
                                     self.n_out, 
                                     device=x.device, 
                                     dtype=x.dtype)

        for block in self.blocks_list:
            # pass input through block
            x_hat, y_hat = block(x)

            # add block forecast to stack forecast
            stack_forecast = stack_forecast + y_hat

            # subtract backcast from input to produce residual
            x = x - x_hat

        stack_residual = x

        return stack_residual, stack_forecast


class NBEATS(nn.Module):
    '''
    N-BEATS
    -------
    refer: 
    1. _TrendGenerator, _SeasonalityGenerator, _Block, _Stack, _NBEATSModule ****
    in darts-master\darts\models\forecasting\nbeats.py
    '''
    def __init__(self, 
                 n_in: int = 8,
                 n_out: int = 1,
                 input_size: int = 3,
                 target_size: int = 3,
                 generic_architecture: bool = True,
                 num_stacks: int = 20, # 30
                 num_blocks: int = 1,
                 num_layers: int = 4,
                 layer_widths: Union[int, List[int]] = 128, # 256
                 expansion_coefficient_dim: int = 5,
                 trend_polynomial_degree: int = 2,
                 ):
        super(NBEATS, self).__init__()
        if not generic_architecture:
            self.num_stacks = 2

        if isinstance(layer_widths, int):
            self.layer_widths = [layer_widths] * num_stacks
        
        self.input_size = input_size
        self.target_size = target_size
        self.n_in_multi = n_in * input_size
        self.n_out = n_out
        self.target_length = n_out * target_size # 'fusing2'

        if generic_architecture:
            self.stacks_list = [
                NbeatsStack(num_blocks,
                       num_layers,
                       self.layer_widths[i],
                       expansion_coefficient_dim,
                       self.n_in_multi,
                       self.target_length,
                       _GType.GENERIC)#.cuda()
                for i in range(num_stacks)
            ]
        else:
            num_stacks = 2
            trend_stack = NbeatsStack(num_blocks,
                                 num_layers,
                                 layer_widths[0],
                                 trend_polynomial_degree + 1,
                                 self.n_in_multi,
                                 self.target_length,
                                 _GType.TREND)#.cuda()
            seasonality_stack = NbeatsStack(num_blocks,
                                       num_layers,
                                       layer_widths[1],
                                       -1,
                                       self.n_in_multi,
                                       self.target_length,
                                       _GType.SEASONALITY)#.cuda()
            self.stacks_list = [trend_stack, seasonality_stack]

        self.stacks = nn.ModuleList(self.stacks_list)

        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)
        self.stacks_list[-1].blocks[-1].backcast_g.requires_grad_(False)

    def forward_nbeats(self, x):

        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc
        x = torch.reshape(x, (x.shape[0], self.n_in_multi, 1))
        # squeeze last dimension (because model is univariate)
        x = x.squeeze(dim=2)

        # One vector of length target_length per parameter in the distribution
        y = torch.zeros(x.shape[0], 
                        self.target_length,
                        device=x.device, 
                        dtype=x.dtype)

        for stack in self.stacks: #stacks_list:
            # compute stack output
            stack_residual, stack_forecast = stack(x)

            # add stack forecast to final output
            y = y + stack_forecast

            # set current stack residual as input for next stack
            x = stack_residual

        # In multivariate case, we get a result [x1_param1, x1_param2], [y1_param1, y1_param2], [x2..], [y2..], ... 
        # We want to reshape to original format. We also get rid of the covariates and keep only the target dimensions.
        # The covariates are by construction added as extra time series on the right side. So we need to get rid of this
        # right output (keeping only :self.target_size).
        y = y.view(y.shape[0], self.n_out, self.target_size)#[:, :, :self.target_size]

        return y
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        return self.forward_nbeats(history_data[...,0]).unsqueeze(-1)