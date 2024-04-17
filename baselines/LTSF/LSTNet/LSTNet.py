import torch
from torch import nn
import torch.nn.functional as F

class LSTNet(nn.Module):
    def __init__(self, n_in=8, n_out=1, input_size=3, target_size=3, hidRNN=100, hidCNN=100, hidSkip=5, CNN_kernel=6, skip=5, highway_window=5, output_fun='sigmoid'):
        super(LSTNet, self).__init__()
        # self.use_cuda = args.cuda
        self.P = n_in # args.window; #'window size'
        self.n_out = n_out
        self.m = input_size # data.m # n_timeseries
        self.target_size = target_size
        self.hidR = hidRNN # args.hidRNN; #'number of RNN hidden units'
        self.hidC = hidCNN # args.hidCNN;
        self.hidS = hidSkip # args.hidSkip;
        self.Ck = CNN_kernel #args.CNN_kernel;
        self.skip = skip #args.skip;
        self.pt = (self.P - self.Ck)//self.skip
        self.hw = highway_window #args.highway_window #'The window size of the highway component'
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))#;
        self.GRU1 = nn.GRU(self.hidC, self.hidR)#;
        self.dropout = nn.Dropout(p=0)#;#dropout applied to layers (0 = no dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, target_size * n_out)
        else:
            self.linear1 = nn.Linear(self.hidR, target_size * n_out) # 'fusing2'
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, n_out)
        self.output = None;
        if (output_fun == 'sigmoid'):
            self.output = torch.sigmoid#;args.
        if (output_fun == 'tanh'):
            self.output = torch.tanh#;args.
 
    def forward_lstnet(self, x):
        batch_size = x.size(0)

        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        res = res.view(batch_size, self.n_out, self.target_size)

        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.n_out, self.target_size)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        return res
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        return self.forward_lstnet(history_data[...,0]).unsqueeze(-1)