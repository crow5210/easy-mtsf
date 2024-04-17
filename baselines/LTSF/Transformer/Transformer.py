from torch import nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Transformer(nn.Module):
    '''
    Transformer Model
    -----------------
    refer: 
    1. _PositionalEncoding, _TransformerModule ****
    in darts-master\darts\models\forecasting\transformer_model.py
    '''
    def __init__(self, n_in: int = 8, n_out: int = 1, input_size: int = 3, target_size: int = 3, d_model: int = 128, nhead: int = 4, num_encoder_layers: int = 3, num_decoder_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0., activation: str = 'relu', custom_encoder = None, custom_decoder = None):
        super(Transformer, self).__init__() #512

        self.input_size = input_size
        self.target_size = target_size
        self.n_out = n_out

        self.encoder = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, n_in)

        # Defining the Transformer module
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation,
                                          custom_encoder=custom_encoder,
                                          custom_decoder=custom_decoder)

        self.decoder = nn.Linear(d_model, n_out * self.target_size) # fusing2

    def _create_transformer_inputs(self, data):
        # '_TimeSeriesSequentialDataset' stores time series in the
        # (batch_size, n_in, input_size) format. PyTorch's nn.Transformer
        # module needs it the (n_in, batch_size, input_size) format.
        # Therefore, the first two dimensions need to be swapped.
        src = data.permute(1, 0, 2)
        tgt = src[-1:, :, :]

        return src, tgt

    def forward_xformer(self, data):
        # Here we create 'src' and 'tgt', the inputs for the encoder and decoder
        # side of the Transformer architecture
        src, tgt = self._create_transformer_inputs(data)

        # "math.sqrt(self.input_size)" is a normalization factor
        # see section 3.2.1 in 'Attention is All you Need' by Vaswani et al. (2017)
        src = self.encoder(src) * math.sqrt(self.input_size)
        src = self.positional_encoding(src)

        tgt = self.encoder(tgt) * math.sqrt(self.input_size)
        tgt = self.positional_encoding(tgt)

        x = self.transformer(src=src, tgt=tgt)
        out = self.decoder(x)

        # Here we change the data format
        # from (1, batch_size, n_out * target_size)
        # to (batch_size, n_out, target_size)
        predictions = out[0, :, :]
        predictions = predictions.view(-1, self.n_out, self.target_size)

        return predictions
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        return self.forward_xformer(history_data[...,0]).unsqueeze(-1)