import types,hashlib
from torch import nn
import torch
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def config_str(cfg, indent = ''):
    s = ''
    for k, v in cfg.items():
        if isinstance(v, dict):
            s += (indent + '{}:').format(k) + '\n'
            s += config_str(v, indent + '  ')
        elif isinstance(v, types.FunctionType):
            s += (indent + '{}: {}').format(k, v.__name__) + '\n'
        else:
            s += (indent + '{}: {}').format(k, v) + '\n'
    return s

def config_md5(cfg_general,cfg_dataset,cfg_model):
    cfg = {**cfg_general,**cfg_dataset,**cfg_model}
    m = hashlib.md5()
    m.update(config_str(cfg).encode('utf-8'))
    return m.hexdigest()


class TSDecoder(nn.Module):
    '''
    an empirical decoder, aiming at the last step

    RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    '''
    def __init__(self, n_in: int = 8, n_out: int = 1, input_size: int = 3, target_size: int = 3, dec_mode: str = 'fusing2'):
        #n_in=timesteps, n_out=prediction_horizon, input_size=n_timeseries, target_size=n_timeseries
        super().__init__()
        self.dec_mode = dec_mode
        self.n_out = n_out
        self.target_size = target_size
        if dec_mode == 'fusing1': ### assuming input_size == target_size by default
            self.fc = nn.Linear(n_in, n_out)#, bias=True
        elif dec_mode == 'fusing2':
            self.fc = nn.Linear(n_in * input_size, n_out * target_size)
        else:
            raise NotImplementedError

    def forward(self, x):  # x: batch_size x timesteps x input_size
        batch_size = x.size(0)

        if self.dec_mode == 'slicing':
            output = x[:, -self.n_out:, :self.target_size] # the worst 
        elif self.dec_mode == 'fusing1':
            output = self.fc(x.permute(0,2,1)).permute(0,2,1)[:, :, :self.target_size]
        elif self.dec_mode == 'fusing2':# view, reshape
            output = self.fc(x.reshape(batch_size, -1)).view(-1, self.n_out, self.target_size)

        return output
    

def data_transformation_4_xformer(history_data: torch.Tensor, future_data: torch.Tensor, start_token_len: int):
    """Transfer the data into the XFormer format.

    Args:
        history_data (torch.Tensor): history data with shape: [B, L1, N, C].
        future_data (torch.Tensor): future data with shape: [B, L2, N, C]. 
                                    L1 and L2 are input sequence length and output sequence length, respectively.
        start_token_length (int): length of the decoder start token. Ref: Informer paper.

    Returns:
        torch.Tensor: x_enc, input data of encoder (without the time features). Shape: [B, L1, N]
        torch.Tensor: x_mark_enc, time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
        torch.Tensor: x_dec, input data of decoder. Shape: [B, start_token_length + L2, N]
        torch.Tensor: x_mark_dec, time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
    """

    # get the x_enc
    x_enc = history_data[..., 0]            # B, L1, N
    # get the corresponding x_mark_enc
    # following previous works, we re-scale the time features from [0, 1) to to [-0.5, 0.5).
    x_mark_enc = history_data[:, :, 0, 1:] - 0.5    # B, L1, C-1

    # get the x_dec
    if start_token_len == 0:
        x_dec = torch.zeros_like(future_data[..., 0])     # B, L2, N
        # get the corresponding x_mark_dec
        x_mark_dec = future_data[..., :, 0, 1:] - 0.5                 # B, L2, C-1
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    else:
        x_dec_token = x_enc[:, -start_token_len:, :]            # B, start_token_length, N
        x_dec_zeros = torch.zeros_like(future_data[..., 0])     # B, L2, N
        x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)    # B, (start_token_length+L2), N
        # get the corresponding x_mark_dec
        x_mark_dec_token = x_mark_enc[:, -start_token_len:, :]            # B, start_token_length, C-1
        x_mark_dec_future = future_data[..., :, 0, 1:] - 0.5          # B, L2, C-1
        x_mark_dec = torch.cat([x_mark_dec_token, x_mark_dec_future], dim=1)    # B, (start_token_length+L2), C-1

    return x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float()