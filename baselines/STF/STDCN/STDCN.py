import torch
from torch import nn
import torch.nn.functional as F
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class series_decomp2(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp2, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        B,D,N,L = x.shape
        x = x.reshape(B*D,N,L).transpose(1,2)
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res.transpose(1,2).reshape(B,D,N,L), moving_mean.transpose(1,2).reshape(B,D,N,L)

class DataEmbedding2(nn.Module):
    def __init__(self, c_in, d_model, num_nodes=None, time_of_day_size=None, day_of_week_size=None, dropout=0.1, fusion_type="SUM"):
        super(DataEmbedding2, self).__init__()
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.num_nodes = num_nodes
        self.fusion_type = fusion_type

        self.value_emb = nn.Conv2d(in_channels=c_in, out_channels=d_model, kernel_size=(1,1), bias=True)

        if self.num_nodes is not None:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, d_model))
            nn.init.xavier_uniform_(self.node_emb)
        if self.time_of_day_size is not None:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, d_model))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.day_of_week_size is not None:
            self.day_in_week_emb = nn.Parameter( torch.empty(self.day_of_week_size, d_model))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        
        self.dropout = nn.Dropout(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

        
    def forward(self, x, x_mark):
        bs,step = x.shape[0],x.shape[1]

        value_embed = self.value_emb(x.unsqueeze(-1).transpose(1,3))

        if self.fusion_type == "SUM":
            if self.time_of_day_size is not None:
                day_embed = self.time_in_day_emb[(x_mark[...,0]*self.time_of_day_size).type(torch.LongTensor)].transpose(1,3)
            else:
                day_embed = 0
            if self.day_of_week_size is not None:
                week_embed = self.day_in_week_emb[(x_mark[...,1]*self.day_of_week_size).type(torch.LongTensor)].transpose(1,3)
            else:
                week_embed = 0
            if self.num_nodes is not None:
                node_embed = self.node_emb.unsqueeze(0).unsqueeze(-1).repeat(bs,1,1,step).transpose(1,2)
            else:
                node_embed = 0
            x = value_embed + day_embed + week_embed + node_embed
        else:
            if self.time_of_day_size is not None:
                day_embed = self.time_in_day_emb[(x_mark[:,-1,:,0]*self.time_of_day_size).type(torch.LongTensor)].transpose(1,2).unsqueeze(-1)
            else:
                day_embed = None
            if self.day_of_week_size is not None:
                week_embed = self.day_in_week_emb[(x_mark[:,-1,:,1]*self.day_of_week_size).type(torch.LongTensor)].transpose(1,2).unsqueeze(-1)
            else:
                week_embed = None
            if self.num_nodes is not None:
                node_embed = self.node_emb.unsqueeze(0).expand(bs, -1, -1).transpose(1, 2).unsqueeze(-1)
            else:
                node_embed = None
            x = torch.cat([value_embed,day_embed,week_embed,node_embed],-1)

        return self.dropout(x)

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim, dilation_size) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 13), dilation=1, padding="same",bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 13), dilation=1, padding="same",bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data                           # residual
        return hidden
    

class STDCN(nn.Module):
    def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
                    dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
                    kernel_size=12, blocks=4, layers=2):
        super(STDCN, self).__init__()

        self.num_nodes = 7
        self.node_dim = 32
        self.input_len = 336
        self.input_dim = 1
        self.embed_dim = 32
        self.output_len = 96
        self.num_layer = 3
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32
        self.time_of_day_size = 96
        self.day_of_week_size = 7

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True


        self.blocks = 3
        self.decom = series_decomp2(25)

        self.emb_layer = DataEmbedding2(in_dim,self.embed_dim,num_nodes,time_in_day,day_in_week,0.05,"SUM")
        self.block_convs = nn.ModuleList()
        for i in range(blocks):
            self.block_convs.append(MultiLayerPerceptron(self.embed_dim, self.embed_dim,i*2))

        self.regression_layer_season = nn.Conv2d(in_channels=self.embed_dim, out_channels=1, kernel_size=(1, 1), bias=True)
        
        self.FC = nn.Linear(self.input_len,self.output_len)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        x_mark = history_data[...,1:3]-0.5
        x = self.emb_layer(history_data[...,0],x_mark)
        residual_trend = 0
        for i in range(self.blocks):
            x = self.block_convs[i](x)
            senson,trend = self.decom(x)
            x = senson
            residual_trend += trend

        prediction = self.regression_layer_season(x+residual_trend)
        return self.FC(prediction).transpose(1,3)
    
# class STDCN(nn.Module):
#     def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
#                     dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
#                     kernel_size=12, blocks=4, layers=2):
#         super(STDCN, self).__init__()
#         self.blocks = blocks
#         self.layers = layers

#         self.senson_convs = nn.ModuleList()
#         self.senson_residual_convs = nn.ModuleList()
#         self.senson_skip_convs = nn.ModuleList()
#         self.senson_bn = nn.ModuleList()

#         self.trend_convs = nn.ModuleList()
#         self.trend_residual_convs = nn.ModuleList()
#         self.trend_skip_convs = nn.ModuleList()
#         self.trend_bn = nn.ModuleList()
#         fution_type = "SUM"


#         self.decom = series_decomp(25)
#         self.senson_emb = DataEmbedding2(in_dim,residual_channels,num_nodes,time_in_day,day_in_week,0.05,fution_type)

#         self.trend_emb = DataEmbedding2(in_dim,residual_channels,num_nodes,time_in_day,day_in_week,0.05,fution_type)

#         receptive_field = 1

#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.senson_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.senson_residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.senson_skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.senson_bn.append(nn.BatchNorm2d(residual_channels))

#                 self.trend_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.trend_residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.trend_skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.trend_bn.append(nn.BatchNorm2d(residual_channels))


#                 new_dilation *= 2
#                 receptive_field += additional_scope
#                 additional_scope *= 2

#         self.end_conv = nn.Conv2d(in_channels=skip_channels,
#                                     out_channels=1,
#                                     kernel_size=(1, 1),
#                                     bias=True)
        
#         in_len = input_len if fution_type == "SUM" else input_len+3
#         self.end_fc = nn.Linear(in_len-kernel_size+1,output_len)


#         self.receptive_field = receptive_field
#         print(self.receptive_field)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         senson,trend = self.decom(history_data[...,0])
#         x_mark = history_data[...,1:3]-0.5
            
#         senson = self.senson_emb(senson,x_mark)
#         trend = self.trend_emb(trend,x_mark)
#         skip_s = 0
#         skip_t = 0

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):
#             residual_s = senson
#             residual_t = trend

#             senson = self.senson_convs[i](senson)
#             trend = self.trend_convs[i](trend)

#             ss = self.senson_skip_convs[i](senson)
#             st = self.trend_skip_convs[i](trend)
#             try:
#                 skip_s += ss
#                 skip_t += st
#             except:
#                 skip_s += nn.functional.pad(ss, (skip_s.shape[-1]-ss.shape[-1], 0, 0, 0))
#                 skip_t += nn.functional.pad(st, (skip_t.shape[-1]-st.shape[-1], 0, 0, 0))

#             senson = self.senson_residual_convs[i](senson)
#             trend = self.trend_residual_convs[i](trend)

#             senson = senson + residual_s[:, :, :, -senson.size(3):]
#             trend = trend + residual_t[:, :, :, -trend.size(3):]

#             senson = self.senson_bn[i](senson)
#             trend = self.trend_bn[i](trend)

#         x = F.relu(skip_s+skip_t)
#         x = F.relu(self.end_conv(x))
#         x = self.end_fc(x)
#         return x.transpose(1,3)
    

    
# DataEmbedding2
# class STDCN(nn.Module):
#     def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
#                     dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
#                     kernel_size=12, blocks=4, layers=2):
#         super(STDCN, self).__init__()

#         self.num_nodes = 7
#         self.node_dim = 32
#         self.input_len = 336
#         self.input_dim = 1
#         self.embed_dim = 32
#         self.output_len = 96
#         self.num_layer = 3
#         self.temp_dim_tid = 32
#         self.temp_dim_diw = 32
#         self.time_of_day_size = 96
#         self.day_of_week_size = 7

#         self.if_time_in_day = True
#         self.if_day_in_week = True
#         self.if_spatial = True


#         # embedding layer
#         self.time_series_emb_layer = DataEmbedding2(in_dim,residual_channels,num_nodes,time_in_day,day_in_week,0.05,"SUM")
        

#         self.encoder = nn.Sequential(
#             *[MultiLayerPerceptron(self.input_len, self.input_len) for _ in range(self.num_layer)])

#         # regression
#         self.regression_layer = nn.Conv2d(
#             in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        
#         self.FC = nn.Linear(residual_channels,1)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         input_data = history_data[...,0]
#         x_mark = history_data[...,1:3]-0.5

            
#         time_series_emb = self.time_series_emb_layer(input_data,x_mark).transpose(1,3)
        

#         # encoding
#         hidden = self.encoder(time_series_emb)

#         # regression
#         prediction = self.regression_layer(hidden)

#         return self.FC(prediction)
    
    

# # stid + cnn 
# # baseline
# class STDCN(nn.Module):
#     def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
#                     dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
#                     kernel_size=12, blocks=4, layers=2):
#         super(STDCN, self).__init__()

#         self.num_nodes = 7
#         self.node_dim = 32
#         self.input_len = 336
#         self.input_dim = 1
#         self.embed_dim = 16
#         self.output_len = 96
#         self.num_layer = 3
#         self.temp_dim_tid = 32
#         self.temp_dim_diw = 32
#         self.time_of_day_size = 96
#         self.day_of_week_size = 7

#         self.if_time_in_day = True
#         self.if_day_in_week = True
#         self.if_spatial = True


#         # embedding layer
#         self.time_series_emb_layer = DataEmbedding2(in_dim,self.embed_dim,num_nodes,time_in_day,day_in_week,0.05,"SUM")
        

#         self.encoder = nn.Sequential(
#             *[MultiLayerPerceptron(self.embed_dim, self.embed_dim,i) for i in range(self.num_layer)])

#         # regression
#         self.regression_layer = nn.Conv2d(
#             in_channels=self.embed_dim, out_channels=1, kernel_size=(1, 1), bias=True)
        
#         self.FC = nn.Linear(self.input_len,self.output_len)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         input_data = history_data[...,0]
#         x_mark = history_data[...,1:3]-0.5

            
#         time_series_emb = self.time_series_emb_layer(input_data,x_mark)
        

#         # encoding
#         hidden = self.encoder(time_series_emb)

#         # regression
#         prediction = self.regression_layer(hidden)

#         return self.FC(prediction).transpose(1,3)

# # stid + cnn + decomp
# class STDCN(nn.Module):
#     def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
#                     dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
#                     kernel_size=12, blocks=4, layers=2):
#         super(STDCN, self).__init__()

#         self.num_nodes = 7
#         self.node_dim = 32
#         self.input_len = 336
#         self.input_dim = 1
#         self.embed_dim = 8
#         self.output_len = 96
#         self.num_layer = 3
#         self.temp_dim_tid = 32
#         self.temp_dim_diw = 32
#         self.time_of_day_size = 96
#         self.day_of_week_size = 7

#         self.if_time_in_day = True
#         self.if_day_in_week = True
#         self.if_spatial = True


#         self.decom = series_decomp(25)

#         self.emb_layer_season = DataEmbedding2(in_dim,self.embed_dim,num_nodes,time_in_day,day_in_week,0.05,"SUM")
#         self.encoder_season = nn.Sequential(*[MultiLayerPerceptron(self.embed_dim, self.embed_dim,i*2) for i in range(self.num_layer)])
#         self.regression_layer_season = nn.Conv2d(in_channels=self.embed_dim, out_channels=1, kernel_size=(1, 1), bias=True)
        
#         self.emb_layer_trend = DataEmbedding2(in_dim,self.embed_dim,num_nodes,time_in_day,day_in_week,0.05,"SUM")
#         # self.encoder_trend = nn.Sequential(*[MultiLayerPerceptron(self.embed_dim, self.embed_dim,i*2) for i in range(self.num_layer)])
#         # self.regression_layer_trend = nn.Conv2d(in_channels=self.embed_dim, out_channels=1, kernel_size=(1, 1), bias=True)
        
#         self.FC = nn.Linear(self.input_len,self.output_len)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         senson,trend = self.decom(history_data[...,0])
#         x_mark = history_data[...,1:3]-0.5
            
#         senson = self.emb_layer_season(senson,x_mark)
#         trend = self.emb_layer_trend(trend,x_mark)
        
        

#         hidden_season = self.encoder_season(senson)
#         prediction_season = self.regression_layer_season(hidden_season)

#         # hidden_trend = self.encoder_trend(trend)
#         # prediction_trend = self.regression_layer_trend(hidden_trend)

#         return self.FC(prediction_season + trend).transpose(1,3)

# # 分解
# class STDCN(nn.Module):
#     def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
#                     dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
#                     kernel_size=12, blocks=4, layers=2):
#         super(STDCN, self).__init__()

#         self.num_nodes = 7
#         self.node_dim = 32
#         self.input_len = 336
#         self.input_dim = 1
#         self.embed_dim = 32
#         self.output_len = 96
#         self.num_layer = 3
#         self.temp_dim_tid = 32
#         self.temp_dim_diw = 32
#         self.time_of_day_size = 96
#         self.day_of_week_size = 7

#         self.if_time_in_day = True
#         self.if_day_in_week = True
#         self.if_spatial = True

#         self.decom = series_decomp(25)
#         self.senson_emb = DataEmbedding2(in_dim,residual_channels,num_nodes,time_in_day,day_in_week,0.05,"SUM")
#         self.encoder_season = nn.Sequential(*[MultiLayerPerceptron(self.input_len, self.input_len) for _ in range(self.num_layer)])
#         self.regression_layer_season = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

#         self.trend_emb = DataEmbedding2(in_dim,residual_channels,num_nodes,time_in_day,day_in_week,0.05,"SUM")
#         self.encoder_trend = nn.Sequential(*[MultiLayerPerceptron(self.input_len, self.input_len) for _ in range(self.num_layer)])
#         self.regression_layer_trend = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

#         self.FC = nn.Linear(residual_channels,1)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         senson,trend = self.decom(history_data[...,0])
#         x_mark = history_data[...,1:3]-0.5

            
#         season_emb = self.senson_emb(senson,x_mark).transpose(1,3)
#         hidden_season = self.encoder_season(season_emb)
#         prediction_season = self.regression_layer_season(hidden_season)

#         trend_emb = self.trend_emb(trend,x_mark).transpose(1,3)
#         hidden_trend = self.encoder_trend(trend_emb)
#         prediction_trend = self.regression_layer_trend(hidden_trend)

#         return self.FC(prediction_season+prediction_trend)
    
# # STDCN 不分解
# class STDCN(nn.Module):
#     def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
#                     dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
#                     kernel_size=12, blocks=4, layers=2):
#         super(STDCN, self).__init__()
#         self.blocks = blocks
#         self.layers = layers

#         self.senson_convs = nn.ModuleList()
#         self.senson_residual_convs = nn.ModuleList()
#         self.senson_skip_convs = nn.ModuleList()
#         self.senson_bn = nn.ModuleList()


#         fution_type = "SUM"
#         self.decom = series_decomp(25)
#         self.senson_emb = DataEmbedding2(in_dim,residual_channels,num_nodes,time_in_day,day_in_week,0.05,fution_type)


#         receptive_field = 1

#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.senson_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.senson_residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.senson_skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.senson_bn.append(nn.BatchNorm2d(residual_channels))


#                 new_dilation *= 2
#                 receptive_field += additional_scope
#                 additional_scope *= 2

#         self.end_conv = nn.Conv2d(in_channels=skip_channels,
#                                     out_channels=1,
#                                     kernel_size=(1, 1),
#                                     bias=True)
        
#         in_len = input_len if fution_type == "SUM" else input_len+3
#         self.end_fc = nn.Linear(in_len-kernel_size+1,output_len)


#         self.receptive_field = receptive_field
#         print(self.receptive_field)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         senson = history_data[...,0]
#         x_mark = history_data[...,1:3]-0.5
            
#         senson = self.senson_emb(senson,x_mark)
#         skip_s = 0

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):
#             residual_s = senson

#             senson = self.senson_convs[i](senson)

#             ss = self.senson_skip_convs[i](senson)
#             try:
#                 skip_s += ss
#             except:
#                 skip_s += nn.functional.pad(ss, (skip_s.shape[-1]-ss.shape[-1], 0, 0, 0))

#             senson = self.senson_residual_convs[i](senson)

#             senson = senson + residual_s[:, :, :, -senson.size(3):]

#             senson = self.senson_bn[i](senson)

#         x = F.relu(skip_s)
#         x = F.relu(self.end_conv(x))
#         x = self.end_fc(x)
#         return x.transpose(1,3)
    

# # STDCN trend不进行卷积
# class STDCN(nn.Module):
#     def __init__(self, in_dim=1, input_len=12, output_len=12, residual_channels=32,
#                     dilation_channels=32, skip_channels=64, num_nodes=170, time_in_day=288, day_in_week=7,
#                     kernel_size=12, blocks=4, layers=2):
#         super(STDCN, self).__init__()
#         self.blocks = blocks
#         self.layers = layers

#         self.senson_convs = nn.ModuleList()
#         self.senson_residual_convs = nn.ModuleList()
#         self.senson_skip_convs = nn.ModuleList()
#         self.senson_bn = nn.ModuleList()


#         fution_type = "SUM"
#         self.decom = series_decomp(25)
#         self.senson_emb = DataEmbedding2(in_dim,residual_channels,num_nodes,time_in_day,day_in_week,0.05,fution_type)
#         self.input_trend = nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1, kernel_size))


#         receptive_field = 1

#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.senson_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.senson_residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.senson_skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.senson_bn.append(nn.BatchNorm2d(residual_channels))


#                 new_dilation *= 2
#                 receptive_field += additional_scope
#                 additional_scope *= 2

#         self.end_conv = nn.Conv2d(in_channels=skip_channels,
#                                     out_channels=1,
#                                     kernel_size=(1, 1),
#                                     bias=True)
        
#         in_len = input_len if fution_type == "SUM" else input_len+3
#         self.end_fc = nn.Linear(in_len-kernel_size+1,output_len)


#         self.receptive_field = receptive_field
#         print(self.receptive_field)

#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         senson,trend = self.decom(history_data[...,0])
#         x_mark = history_data[...,1:3]-0.5
            
#         senson = self.senson_emb(senson,x_mark)
#         trend = self.senson_emb(trend,x_mark)
#         skip_s = 0

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):
#             residual_s = senson

#             senson = self.senson_convs[i](senson)

#             ss = self.senson_skip_convs[i](senson)
#             try:
#                 skip_s += ss
#             except:
#                 skip_s += nn.functional.pad(ss, (skip_s.shape[-1]-ss.shape[-1], 0, 0, 0))

#             senson = self.senson_residual_convs[i](senson)

#             senson = senson + residual_s[:, :, :, -senson.size(3):]

#             senson = self.senson_bn[i](senson)

#         skip_t = self.input_trend(trend)
#         x = F.relu(skip_s + skip_t)
#         x = F.relu(self.end_conv(x))
#         x = self.end_fc(x)
#         return x.transpose(1,3)