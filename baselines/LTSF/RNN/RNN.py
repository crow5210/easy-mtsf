import torch
from torch import nn
from utils.__init__ import TSDecoder

class _RNN(nn.Module):
    '''
    Recurrent Neural Networks
    -------------------------
    refer: 
    1. _RNNModule *****
    in darts-master\darts\models\forecasting\rnn_model.py
    2. RNNModel **
    in gluonts\model\deep_factor\RNNModel.py
    '''
    def __init__(self, name='LSTM', n_in=8, n_out=1, input_size=3, target_size=3, hidden_layer_size=128, num_layers=3, bidirectional=True, dropout=0.):
        super(_RNN, self).__init__() # 'GRU'，'LSTM', 100, 1, False
        self.input_size = input_size
        self.D = 2 if bidirectional else 1

        self.rnn = getattr(nn, name)(input_size=input_size, 
                            hidden_size=hidden_layer_size, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            batch_first=True,
                            dropout=dropout)#.cuda()
        self.decoder = nn.Linear(self.D*hidden_layer_size, target_size)#.cuda() # target_size=1, univariate
        # self.fc = nn.Linear(n_in, n_out)#.cuda()
        ### refer TS2Vec
        self.tsdecoder = TSDecoder(n_in=n_in, n_out=n_out, input_size=input_size, target_size=target_size)#, dec_mode='fusing2'
        

    def forward(self, input_seq):
        batch_size = input_seq.shape[0] # len(input_seq)
        self.rnn.flatten_parameters()
        rnn_out, last_hidden_state = self.rnn(input_seq) #.reshape(batch_size, -1, self.input_size)
        predictions = self.decoder(rnn_out) #.view(batch_size, -1)
        
        return self.tsdecoder(predictions)
        #self.fc(predictions.permute(0,2,1)).permute(0,2,1)

###################################################
# use_cuda = torch.cuda.is_available()

# Cell
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.dropout = dropout

    def forward(self, inputs, hidden):
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)
        gates = (torch.matmul(inputs, self.weight_ih.t()) + self.bias_ih +
                         torch.matmul(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

# Cell
class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMCell, self).__init__()
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        self.dropout = dropout

    def forward(self, inputs, hidden):
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)

        ifo_gates = (torch.matmul(inputs, self.weight_ii.t()) + self.bias_ii +
                                  torch.matmul(hx, self.weight_ih.t()) + self.bias_ih +
                                  torch.matmul(cx, self.weight_ic.t()) + self.bias_ic)
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)

        cellgate = torch.matmul(hx, self.weight_hh.t()) + self.bias_hh

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)

        if self.input_size == self.hidden_size:
            hy = outgate * (ry + inputs)
        else:
            hy = outgate * (ry + torch.matmul(inputs, self.weight_ir.t()))
        return hy, (hy, cy)

# Cell
class ResLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ResLSTMCell(input_size, hidden_size, dropout=0.)

    def forward(self, inputs, hidden):
        inputs = inputs.unbind(0)
        outputs = []
        for i in range(len(inputs)):
                out, hidden = self.cell(inputs[i], hidden)
                outputs += [out]
        outputs = torch.stack(outputs)
        return outputs, hidden

# Cell
class AttentiveLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(AttentiveLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        attention_hsize = hidden_size
        self.attention_hsize = attention_hsize

        self.cell = LSTMCell(input_size, hidden_size)
        self.attn_layer = nn.Sequential(nn.Linear(2 * hidden_size + input_size, attention_hsize),
                                        nn.Tanh(),
                                        nn.Linear(attention_hsize, 1))
        self.softmax = nn.Softmax(dim=0)
        self.dropout = dropout

    def forward(self, inputs, hidden):
        inputs = inputs.unbind(0)
        outputs = []

        for t in range(len(inputs)):
            # attention on windows
            hx, cx = (tensor.squeeze(0) for tensor in hidden)
            hx_rep = hx.repeat(len(inputs), 1, 1)
            cx_rep = cx.repeat(len(inputs), 1, 1)
            x = torch.cat((inputs, hx_rep, cx_rep), dim=-1)
            l = self.attn_layer(x)
            beta = self.softmax(l)
            context = torch.bmm(beta.permute(1, 2, 0),
                                inputs.permute(1, 0, 2)).squeeze(1)
            out, hidden = self.cell(context, hidden)
            outputs += [out]
        outputs = torch.stack(outputs)
        return outputs, hidden

# Cell
# class DRNN(nn.Module):
class DilRNN(nn.Module):
    '''
    Dilated Recurrent Neural Networks
    ---------------------------------
    refer: 
    1. DRNN ***
    in pytorch-dilated-rnn-master\drnn.py
    2. DRNN *****
    in n-hits-main\src\models\components\drnn.py
    '''
    def __init__(self, n_input, n_hidden, n_layers, dilations, dropout=0, cell_type='GRU', batch_first=True):
        # dropout=0, batch_first=True, bidirectional=False, cell_type='GRU'):
        super(DilRNN, self).__init__()

        assert n_layers==len(dilations) # [1 for i in range(n_layers)]
        self.dilations = dilations # [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "ResLSTM": # 不行
            cell = ResLSTMLayer
        elif self.cell_type == "AttentiveLSTM": # 也不行。。。
            cell = AttentiveLSTMLayer
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size,
                                                       hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            hidden = torch.zeros(batch_size * rate, hidden_size,
                                 dtype=dilated_inputs.dtype,
                                 device=dilated_inputs.device)
            hidden = hidden.unsqueeze(0)

            if self.cell_type in ['LSTM', 'ResLSTM', 'AttentiveLSTM']:
                hidden = (hidden, hidden)

        dilated_outputs, hidden = cell(dilated_inputs, hidden) # compatibility hack

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2),
                                 dtype=inputs.dtype,
                                 device=inputs.device)
            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs


class RNN(nn.Module):
    '''
    Recurrent Neural Networks
    -------------------------
    refer: 
    1. _RNNModule *****
    in darts-master\darts\models\forecasting\rnn_model.py
    2. RNNModel **
    in gluonts\model\deep_factor\RNNModel.py
    '''
    def __init__(self, name='LSTM', n_in=8, n_out=1, input_size=3, target_size=3, hidden_layer_size=128, n_layers=3, bidirectional=False, dropout=0, cell_type='GRU'):
        super(RNN, self).__init__() # 'GRU', 100, 1, False, num_layers=3, bidirectional=True, dropout=0.
        self.input_size = input_size
        self.D = 2 if bidirectional and name!='DilRNN' else 1
        self.name = name

        self.rnn = DilRNN(n_input=input_size,
                        n_hidden=hidden_layer_size,
                        n_layers=n_layers,
                        dilations=[2 ** i for i in range(n_layers)],
                        # bidirectional=bidirectional, 
                        batch_first=True,
                        dropout=dropout,
                        cell_type=cell_type
                       ) if name=='DilRNN' else getattr(nn, name)(
                            input_size=input_size, 
                            hidden_size=hidden_layer_size, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            batch_first=True,
                            dropout=dropout)
        self.decoder = nn.Linear(self.D*hidden_layer_size, input_size)
        #self.fc = nn.Linear(n_in, n_out)
        ### refer TS2Vec
        self.tsdecoder = TSDecoder(n_in=n_in, n_out=n_out, input_size=input_size, target_size=target_size)#, dec_mode='fusing2'

    def forward_rnn(self, input_seq): # input_seq: batch_size x timesteps x input_size
        batch_size = input_seq.shape[0]
        if self.name in ['RNN','LSTM','GRU']:
            self.rnn.flatten_parameters()
        rnn_out, last_hidden_state = self.rnn(input_seq)
        predictions = self.decoder(rnn_out)
        
        return self.tsdecoder(predictions)
        #self.fc(predictions.permute(0,2,1)).permute(0,2,1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        return self.forward_rnn(history_data[...,0]).unsqueeze(-1)