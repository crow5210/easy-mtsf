import torch
import numpy as np
import scipy.sparse as sp
import math
from easydict import EasyDict
from utils.metrics import masked_mae,masked_mse
from utils.serialization import load_pkl,load_adj
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# -----------ltssf----------------------------
from baselines.LTSF.Autoformer import Autoformer
from baselines.LTSF.Informer import Informer
from baselines.LTSF.DLinear import DLinear
from baselines.LTSF.NLinear import NLinear
from baselines.LTSF.Crossformer import Crossformer
from baselines.LTSF.WaveNet import WaveNet
from baselines.LTSF.FEDformer import FEDformer
from baselines.LTSF.Linear import Linear
from baselines.LTSF.HI import HINetwork
from baselines.LTSF.MLP import MultiLayerPerceptron
from baselines.LTSF.NHiTS import NHiTS
from baselines.LTSF.PatchTST import PatchTST
from baselines.LTSF.Pyraformer import Pyraformer
from baselines.LTSF.WaveNet import WaveNet
from baselines.LTSF.Transformer import Transformer
from baselines.LTSF.TCN import TCN
from baselines.LTSF.NBEATS import NBEATS
from baselines.LTSF.RNN import RNN
from baselines.LTSF.LSTNet import LSTNet
from baselines.LTSF.Triformer import Triformer
from utils.dilate_loss import dilate_loss

# ------------stf------------------------
from baselines.STF.STID import STID
from baselines.STF.STDCN import STDCN
from baselines.STF.AGCRN import AGCRN
from baselines.STF.BGSLF import BGSLF
from baselines.STF.D2STGNN import D2STGNN
from baselines.STF.DGCRN import DGCRN
from baselines.STF.DCRNN import DCRNN
from baselines.STF.GTS import GTS
from baselines.STF.GTS.loss import gts_loss
from baselines.STF.GWNet import GraphWaveNet
from baselines.STF.MegaCRN import MegaCRN
from baselines.STF.MegaCRN.loss import megacrn_loss
from baselines.STF.STAEformer import STAEformer
from baselines.STF.StemGNN import StemGNN
from baselines.STF.STGCN import STGCN
from baselines.STF.STGODE import STGODE
from baselines.STF.STGODE.generate_matrices import generate_dtw_spa_matrix
from baselines.STF.STNorm import STNorm
from baselines.STF.STWave import STWave
from baselines.STF.MTGNN import MTGNN


from baselines.Traditional import Traditional

CFG_MODEL = EasyDict()

# ======== model: Traditional =============== #
def NAIVE_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"NAIVE_{ds_name}"
    MODEL.ARCH = Traditional
    MODEL.UNIVARIATE = True
    MODEL.PARAM = {
        "name":"NAIVE",
        "args":{"K":12}
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.NAIVE = NAIVE_ARGS

def ES_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"ES_{ds_name}"
    MODEL.ARCH = Traditional
    MODEL.UNIVARIATE = True
    MODEL.PARAM = {
        "name":"ES",
        "args":{
            "seasonal_periods":12
        }
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.ES = ES_ARGS

def Theta_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Theta_{ds_name}"
    MODEL.ARCH = Traditional
    MODEL.UNIVARIATE = True
    MODEL.PARAM = {
        "name":"Theta",
        "args":{"theta":1.5}
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.Theta = Theta_ARGS

def ARIMA_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"ARIMA_{ds_name}"
    MODEL.ARCH = Traditional
    MODEL.UNIVARIATE = True
    MODEL.PARAM = {
        "name":"ARIMA",
        "args":{"p":12,"d":1,"q":1}
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.ARIMA = ARIMA_ARGS

def Kalman_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Kalman_{ds_name}"
    MODEL.ARCH = Traditional
    MODEL.UNIVARIATE = True
    MODEL.PARAM = {
        "name":"Kalman",
        "args":{"dim_x":12}
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.Kalman = Kalman_ARGS


def LGBM_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"LGBM_{ds_name}"
    MODEL.ARCH = Traditional
    MODEL.UNIVARIATE = False
    MODEL.PARAM = {
        "name":"LGBM",
        "args":{"lags":35,"output_chunk_length":1,"objective":"mape"}
    }
    return MODEL
CFG_MODEL.LGBM = LGBM_ARGS

def Random_Forest_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Random_Forest_{ds_name}"
    MODEL.ARCH = Traditional
    MODEL.UNIVARIATE = False
    MODEL.PARAM = {
        "name":"Random_Forest",
        "args":{"lags":30,"output_chunk_length":1}
    }
    return MODEL
CFG_MODEL.Random_Forest = Random_Forest_ARGS

# ======== model: AGCRN =============== #
def AGCRN_ARGS(ds_name, num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"AGCRN_{ds_name}"
    MODEL.ARCH = AGCRN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes" : num_nodes,
        "input_dim" : 1,
        "rnn_units" : 64,
        "output_dim": 1,
        "horizon"   : output_len,
        "num_layers": 2,
        "default_graph": True,
        "embed_dim" : 2,
        "cheb_k"    : 2
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL

CFG_MODEL.AGCRN = AGCRN_ARGS


# ======== model: Autoformer =============== #
def Autoformer_ARGS(ds_name, num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Autoformer_{ds_name}"
    MODEL.ARCH = Autoformer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "seq_len": input_len,
        "label_len": input_len/2,       # start token length used in decoder
        "pred_len": output_len,         # prediction sequence length
        "moving_avg": 65,                           # window size of moving average. This is a CRUCIAL hyper-parameter.
        "output_attention": False,
        "enc_in": num_nodes,                        # num nodes
        "dec_in": num_nodes,
        "c_out": num_nodes,
        "d_model": 512,
        "embed": "timeF",                           # [timeF, fixed, learned]
        "dropout": 0.05,
        "factor": 6 if input_len>100 else 3,                                # attn factor
        "n_heads": 8,
        "d_ff": 2048,
        "activation": "gelu",
        "e_layers": 2,                              # num of encoder layers
        "d_layers": 1,                              # num of decoder layers
        "num_time_features": 2,                     # number of used time features
        "time_of_day_size": steps_per_day,
        "day_of_week_size": 7,
        }
    return MODEL
CFG_MODEL.Autoformer = Autoformer_ARGS


# ======== model: BGSLF =============== #
def BGSLF_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"BGSLF_{ds_name}"
    MODEL.ARCH = BGSLF
    MODEL.LOSS = masked_mae
    node_feats_full = load_pkl(f"datasets/{ds_name}/data_in_{input_len}_out_{output_len}_rescale_{rescale}.pkl")["processed_data"][..., 0]
    train_index_list = load_pkl(f"datasets/{ds_name}/index_in_{input_len}_out_{output_len}_rescale_{rescale}.pkl")["train"]
    node_feats = node_feats_full[:train_index_list[-1][-1], ...]
    MODEL.PARAM = {
        "node_feas": torch.Tensor(node_feats),
        "temperature": 0.5,
        "args": EasyDict({
            "device": torch.device("cuda:0"),
            "cl_decay_steps": 2000,
            "filter_type": "dual_random_walk",
            "horizon": output_len,
            "feas_dim": 1,
            "input_dim": 2,
            "ll_decay": 0,
            "num_nodes": num_nodes,
            "max_diffusion_step": 2,
            "num_rnn_layers": 1,
            "output_dim": 1,
            "rnn_units": 64,
            "seq_len": input_len,
            "use_curriculum_learning": True,
            "embedding_size": 256,
            "kernel_size": 12,
            "freq": steps_per_day,
            "requires_graph": 2
        })

    }
    MODEL.FORWARD_FEATURES = [0, 1]
    return MODEL
CFG_MODEL.BGSLF = BGSLF_ARGS





# ======== model: Crossformer =============== #
def Crossformer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Crossformer_{ds_name}"
    MODEL.ARCH = Crossformer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "data_dim": num_nodes,
        "in_len": input_len,
        "out_len": output_len,
        "seg_len": 24,
        "win_size": 2,
        # default parameters
        "factor": 10,
        "d_model": 64,
        "d_ff": 64,
        "n_heads": 4,
        "e_layers": 3,
        "dropout": 0.2,
        "baseline": False
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.Crossformer = Crossformer_ARGS



# ======== model: D2STGNN =============== #
def D2STGNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"D2STGNN_{ds_name}"
    MODEL.ARCH = D2STGNN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "num_feat": 1,
        "num_hidden": 32,
        "dropout": 0.1,
        "seq_length": output_len,
        "k_t": 3,
        "k_s": 2,
        "gap": 3,
        "num_nodes": num_nodes,
        "adjs": [torch.tensor(adj) for adj in adj_mx],
        "num_layers": 5,
        "num_modalities": 2,
        "node_hidden": 10,
        "time_emb_dim": 10,
        "time_in_day_size": steps_per_day,
        "day_in_week_size": 7,
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.D2STGNN = D2STGNN_ARGS


# ======== model: DCRNN =============== #
def DCRNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"DCRNN{ds_name}"
    MODEL.ARCH = DCRNN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "cl_decay_steps": 2000,
        "horizon": output_len,
        "input_dim": 2,
        "max_diffusion_step": 2,
        "num_nodes": num_nodes,
        "num_rnn_layers": 2,
        "output_dim": 1,
        "rnn_units": 64,
        "seq_len": input_len,
        "adj_mx": [torch.tensor(i) for i in adj_mx],
        "use_curriculum_learning": True
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.DCRNN = DCRNN_ARGS

# ======== model: DGCRN =============== #
def DGCRN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"DGCRN_{ds_name}"
    MODEL.ARCH = DGCRN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "gcn_depth": 2,
        "num_nodes": num_nodes,
        "predefined_A": [torch.Tensor(_) for _ in adj_mx],
        "dropout": 0.3,
        "subgraph_size": 20,
        "node_dim": 40,
        "middle_dim": 2,
        "seq_length": input_len,
        "in_dim": 2,
        "list_weight": [0.05, 0.95, 0.95],
        "tanhalpha": 3,
        "cl_decay_steps": 4000,
        "rnn_size": 64,
        "hyperGNN_dim": 16
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.DGCRN = DGCRN_ARGS


# ======== model: DLinear =============== #
def DLinear_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"DLinear_{ds_name}"
    MODEL.ARCH = DLinear
    MODEL.LOSS = masked_mse
    MODEL.PARAM = {
        "seq_len": input_len,
        "pred_len": output_len,
        "individual": False,
        "enc_in": num_nodes
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.DLinear = DLinear_ARGS



# ======== model: FEDformer =============== #
def FEDformer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"FEDformer_{ds_name}"
    MODEL.ARCH = FEDformer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "enc_in": num_nodes,                        # num nodes
        "dec_in": num_nodes,
        "c_out": num_nodes,
        "seq_len": input_len,           # input sequence length
        "label_len": input_len/2,       # start token length used in decoder
        "pred_len": output_len,         # prediction sequence length\
        "d_model": 512,
        "version": "Fourier",                       # for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]
        "moving_avg": 24,                           # window size of moving average
        "n_heads": 8,
        "e_layers": 2,                              # num of encoder layers
        "d_layers": 1,                               # num of decoder layers
        "d_ff": 2048,
        "dropout": 0.05,
        "output_attention": False,
        "embed": "timeF",                           # [timeF, fixed, learned]
        "mode_select": "random",                    # for FEDformer, there are two mode selection method, options: [random, low]
        "modes": 64,                                # modes to be selected random 64
        "base": "legendre",                         # mwt base
        "L": 3,                                     # ignore level
        "cross_activation": "tanh",                 # mwt cross atention activation function tanh or softmax
        "activation": "gelu",
        "num_time_features": 2,                     # number of used time features
        "time_of_day_size": steps_per_day,
        "day_of_week_size": 7
        }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.FEDformer = FEDformer_ARGS




# ======== model: GTS =============== #
def GTS_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"GTS_{ds_name}"
    MODEL.ARCH = GTS
    MODEL.LOSS = gts_loss
    node_feats_full = load_pkl(f"datasets/{ds_name}/data_in_{input_len}_out_{output_len}_rescale_{rescale}.pkl")["processed_data"][..., 0]
    train_index_list = load_pkl(f"datasets/{ds_name}/index_in_{input_len}_out_{output_len}_rescale_{rescale}.pkl")["train"]
    node_feats = node_feats_full[:train_index_list[-1][-1], ...]
    MODEL.PARAM = {
        "cl_decay_steps": 2000,
        "filter_type": "dual_random_walk",
        "horizon": output_len,
        "input_dim": 2,
        "l1_decay": 0,
        "max_diffusion_step": 3,
        "num_nodes": num_nodes,
        "num_rnn_layers": 1,
        "output_dim": 1,
        "rnn_units": 64,
        "seq_len": input_len,
        "use_curriculum_learning": True,
        "dim_fc": (node_feats.shape[0]-18)*16,
        "node_feats": node_feats,
        "temp": 0.5,
        "k": 30
    }
    MODEL.FORWARD_FEATURES = [0, 1]
    return MODEL
CFG_MODEL.GTS = GTS_ARGS




# ======== model: GraphWaveNet =============== #
def GraphWaveNet_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"GraphWaveNet_{ds_name}"
    MODEL.ARCH = GraphWaveNet
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "num_nodes": num_nodes,
        "supports": [torch.tensor(i) for i in adj_mx],
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": 2,
        "out_dim": output_len,
        "residual_channels": 16,
        "dilation_channels": 16,
        "skip_channels": 64,
        "end_channels": 128,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2
    }
    MODEL.FORWARD_FEATURES = [0, 1]
    return MODEL
CFG_MODEL.GraphWaveNet = GraphWaveNet_ARGS



# ======== model: HINetwork =============== #
def HINetwork_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"HINetwork_{ds_name}"
    MODEL.ARCH = HINetwork
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "input_length": input_len,
        "output_length": output_len,
    }
    MODEL.FORWARD_FEATURES = [0, 1]
    return MODEL
CFG_MODEL.HINetwork = HINetwork_ARGS


# ======== model: Informer =============== #
def Informer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Informer_{ds_name}"
    MODEL.ARCH = Informer
    MODEL.LOSS = masked_mse
    MODEL.PARAM = {
        "enc_in": num_nodes,                              # num nodes
        "dec_in": num_nodes,
        "c_out": num_nodes,
        "seq_len": input_len,           # input sequence length
        "label_len": input_len/2,       # start token length used in decoder
        "out_len": output_len,          # prediction sequence length\
        "factor": 3,                                # probsparse attn factor
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,                              # num of encoder layers
        # "e_layers": [4, 2, 1],                    # for InformerStack
        "d_layers": 1,                              # num of decoder layers
        "d_ff": 2048,
        "dropout": 0.05,
        "attn": 'prob',                             # attention used in encoder, options:[prob, full]
        "embed": "timeF",                           # [timeF, fixed, learned]
        "activation": "gelu",
        "output_attention": False,
        "distil": True,                             # whether to use distilling in encoder, using this argument means not using distilling
        "mix": True,                                # use mix attention in generative decoder
        "num_time_features": 2,                     # number of used time features [time_of_day, day_of_week, day_of_month, day_of_year]
        "time_of_day_size": steps_per_day,
        "day_of_week_size": 7,
        "day_of_month_size": 31,
        "day_of_year_size": 366
        }
    MODEL.FORWARD_FEATURES = [0, 1, 2]
    return MODEL
CFG_MODEL.Informer = Informer_ARGS



# ======== model: Linear =============== #
def Linear_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Linear_{ds_name}"
    MODEL.ARCH = Linear
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "seq_len": input_len,
        "pred_len": output_len,
        "individual": False,
        "enc_in": num_nodes
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.Linear = Linear_ARGS



# ======== model: LSTNet =============== #
def LSTNet_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"LSTNet_{ds_name}"
    MODEL.ARCH = LSTNet
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "n_in":input_len, 
        "n_out":output_len, 
        "input_size":num_nodes, 
        "target_size":num_nodes
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.LSTNet = LSTNet_ARGS

# ======== model: MegaCRN =============== #
def MegaCRN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"MegaCRN_{ds_name}"
    MODEL.ARCH = MegaCRN
    MODEL.LOSS = megacrn_loss
    MODEL.PARAM = {
        "num_nodes": num_nodes,
        "input_dim": 1,
        "output_dim": 1,
        "horizon": output_len,
        "rnn_units": 64,
        "num_layers":1,
        "cheb_k":3,
        "ycov_dim":1,
        "mem_num":20,
        "mem_dim":64,
        "cl_decay_steps":2000,
        "use_curriculum_learning":True
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.MegaCRN = MegaCRN_ARGS




# ======== model: MultiLayerPerceptron =============== #
def MultiLayerPerceptron_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"MultiLayerPerceptron_{ds_name}"
    MODEL.ARCH = MultiLayerPerceptron
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "history_seq_len": input_len,
        "prediction_seq_len": output_len,
        "hidden_dim": 32
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.MultiLayerPerceptron = MultiLayerPerceptron_ARGS


# ======== model: MTGNN =============== #
def MTGNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    buildA_true = True
    if buildA_true: # self-learned adjacency matrix
        adj_mx = None
    else:           # use predefined adjacency matrix
        _, adj_mx = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
        adj_mx = torch.tensor(adj_mx)-torch.eye(num_nodes)

    MODEL = EasyDict()
    MODEL.NAME = f"MTGNN_{ds_name}"
    MODEL.ARCH = MTGNN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "gcn_true"  : True,
        "buildA_true": buildA_true,
        "gcn_depth": 2,
        "num_nodes": num_nodes,
        "predefined_A":adj_mx,
        "dropout":0.3,
        "subgraph_size":20,
        "node_dim":40,
        "dilation_exponential":1,
        "conv_channels":32,
        "residual_channels":32,
        "skip_channels":64,
        "end_channels":128,
        "seq_length":input_len,
        "in_dim":2,
        "out_dim":output_len,
        "layers":3,
        "propalpha":0.05,
        "tanhalpha":3,
        "layer_norm_affline":True
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.MTGNN = MTGNN_ARGS

# ======== model: NBEATS =============== #
def NBEATS_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"NBEATS_{ds_name}"
    MODEL.ARCH = NBEATS
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "n_in":input_len, 
        "n_out":output_len, 
        "input_size":num_nodes, 
        "target_size":num_nodes, 
        "generic_architecture":True, 
        "num_stacks":10, 
        "num_blocks":1, 
        "num_layers":4, 
        "layer_widths":256, 
        "expansion_coefficient_dim":5, 
        "trend_polynomial_degree":2
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.NBEATS = NBEATS_ARGS


# ======== model: NHiTS =============== #
def NHiTS_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"NHiTS_{ds_name}"
    MODEL.ARCH = NHiTS
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "context_length": input_len,
        "prediction_length": output_len,
        "output_size": 1,
        "n_blocks": [1, 1, 1],
        "n_layers": [2, 2, 2, 2, 2, 2, 2, 2],
        "hidden_size": [[512, 512], [512, 512], [512, 512]],
        "pooling_sizes": [8, 8, 8],
        "downsample_frequencies": [24, 12, 1]
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.NHiTS = NHiTS_ARGS



# ======== model: NLinear =============== #
def NLinear_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"NLinear_{ds_name}"
    MODEL.ARCH = NLinear
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "seq_len": input_len,
        "pred_len": output_len,
        "individual": False,
        "enc_in": num_nodes
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.NLinear = NLinear_ARGS




# ======== model: PatchTST =============== #
def PatchTST_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"PatchTST_{ds_name}"
    MODEL.ARCH = PatchTST
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "enc_in": num_nodes,                        # num nodes
        "seq_len": input_len,           # input sequence length
        "pred_len": output_len,         # prediction sequence length
        "e_layers": 3,                              # num of encoder layers
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0.0,
        "patch_len": 16,
        "stride": 8,
        "individual": 0,                            # individual head; True 1 False 0
        "padding_patch": "end",                     # None: None; end: padding on the end
        "revin": 1,                                 # RevIN; True 1 False 0
        "affine": 0,                                # RevIN-affine; True 1 False 0
        "subtract_last": 0,                         # 0: subtract mean; 1: subtract last
        "decomposition": 0,                         # decomposition; True 1 False 0
        "kernel_size": 25,                          # decomposition-kernel
        }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.PatchTST = PatchTST_ARGS



# ======== model: Pyraformer =============== #
def Pyraformer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Pyraformer_{ds_name}"
    MODEL.ARCH = Pyraformer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "enc_in": num_nodes,                        # num nodes
        "dec_in": num_nodes,
        "c_out": num_nodes,
        "input_size": input_len,
        "predict_step": output_len,
        "d_model": 512,
        "d_inner_hid": 512,
        "d_k": 128,
        "d_v": 128,
        "d_bottleneck": 128,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.05,
        "decoder": "FC",                            # FC or attention
        "window_size": "[2, 2, 2]",
        "inner_size": 5,
        "CSCM": "Bottleneck_Construct",
        "truncate": False,
        "use_tvm": False,
        "embed": "DataEmbedding",
        "num_time_features": 2,
        "time_of_day_size": steps_per_day,
        "day_of_week_size": 7,
        }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.Pyraformer = Pyraformer_ARGS


# ======== model: RNN =============== #
def RNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"RNN_{ds_name}"
    MODEL.ARCH = RNN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "n_in":input_len, 
        "n_out":output_len, 
        "input_size":num_nodes, 
        "target_size":num_nodes, 
        "name":'GRU', 
        "hidden_layer_size":100, 
        "bidirectional":False
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.RNN = RNN_ARGS


# ======== model: STAEformer =============== #
def STAEformer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STAEformer_{ds_name}"
    MODEL.ARCH = STAEformer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes" : num_nodes,
        "in_steps":input_len,
        "out_steps":output_len,
        "feed_forward_dim":64,
        "adaptive_embedding_dim":32
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.STAEformer = STAEformer_ARGS



# ======== model: StemGNN =============== #
def StemGNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"StemGNN_{ds_name}"
    MODEL.ARCH = StemGNN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "units": num_nodes,
        "stack_cnt": 2,
        "time_step": input_len,
        "multi_layer": 5,
        "horizon": output_len,
        "dropout_rate": 0.5,
        "leaky_rate": 0.2
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.StemGNN = StemGNN_ARGS





# ======== model: STGCN =============== #
def STGCN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STGCN_{ds_name}"
    MODEL.ARCH = STGCN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "normlap")
    adj_mx = torch.Tensor(adj_mx[0])
    MODEL.PARAM = {
        "Ks" : 3, 
        "Kt" : 3,
        "blocks" : [[1], [64, 16, 64], [64, 16, 64], [128, 128], [output_len]],
        "T" : input_len,
        "n_vertex" : num_nodes,
        "act_func" : "glu",
        "graph_conv_type" : "cheb_graph_conv",
        "gso" : adj_mx,
        "bias": True,
        "droprate" : 0.5
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.STGCN = STGCN_ARGS




# ======== model: STGODE =============== #
def STGODE_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STGODE_{ds_name}"
    MODEL.ARCH = STGODE
    MODEL.LOSS = masked_mae
    A_se_wave, A_sp_wave = generate_dtw_spa_matrix(ds_name, input_len, output_len)
    MODEL.PARAM = {
        "num_nodes": num_nodes,
        "num_features": 3,
        "num_timesteps_input": input_len,
        "num_timesteps_output": output_len,
        "A_sp_hat" : A_sp_wave,
        "A_se_hat" : A_se_wave
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.STGODE = STGODE_ARGS



# ======== model: STID =============== #
def STID_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STID_{ds_name}"
    MODEL.ARCH = STID
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes":num_nodes,
        "input_len": input_len,
        "input_dim": 1,
        "embed_dim": 32,
        "output_len": output_len,
        "num_layer": 3,
        "if_node": True,
        "node_dim": 32,
        "if_T_i_D": True,
        "if_D_i_W": True,
        "temp_dim_tid": 32,
        "temp_dim_diw": 32,
        "time_of_day_size": steps_per_day,
        "day_of_week_size": 7,
        "type":"FC",                # [FC,CONV1D,CONV2D]
        "concat":True
    }
    return MODEL
CFG_MODEL.STID = STID_ARGS


# ======== model: STDCN =============== #
def STDCN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STDCN_{ds_name}"
    MODEL.ARCH = STDCN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes":num_nodes,
        "in_dim":1, 
        "input_len":input_len, 
        "output_len":output_len, 
        "residual_channels":32,
        "dilation_channels":32, 
        "skip_channels":32, 
        "kernel_size":12 if input_len >= 336 else 3, 
        "blocks":3 if input_len >= 336 else 1, 
        "layers":2,
        "time_in_day":steps_per_day,
        "day_in_week":7
    }
    return MODEL
CFG_MODEL.STDCN = STDCN_ARGS




# ======== model: STNorm =============== #
def STNorm_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STNorm_{ds_name}"
    MODEL.ARCH = STNorm
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes" : num_nodes,
        "tnorm_bool": True,
        "snorm_bool": True,
        "in_dim"    : 2,
        "out_dim"   : output_len,
        "channels"  : 32,
        "kernel_size": 2,
        "blocks"    : 4,
        "layers"    : 2,
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.STNorm = STNorm_ARGS




# ======== model: STWave =============== #
def STWave_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    def laplacian(W):
        """Return the Laplacian of the weight matrix."""
        # Degree matrix.
        d = W.sum(axis=0)
        # Laplacian matrix.
        d = 1 / np.sqrt(d)
        D = sp.diags(d, 0)
        I = sp.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
        return L

    def largest_k_lamb(L, k):
        lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
        return (lamb, U)

    def get_eigv(adj,k):
        L = laplacian(adj)
        eig = largest_k_lamb(L,k)
        return eig

    def loadGraph(adj_mx, hs, ls):
        graphwave = get_eigv(adj_mx+np.eye(adj_mx.shape[0]), hs)
        sampled_nodes_number = int(np.around(math.log(adj_mx.shape[0]))+2)*ls
        graph = csr_matrix(adj_mx)
        dist_matrix = dijkstra(csgraph=graph)
        dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
        adj_gat = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
        return adj_gat, graphwave

    MODEL = EasyDict()
    MODEL.NAME = f"STWave_{ds_name}"
    MODEL.ARCH = STWave
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name +  "/adj_mx.pkl", "original")
    adjgat, gwv = loadGraph(_, 128, 1)
    MODEL.PARAM = {
        "input_dim": 1,
        "hidden_size": 128,
        "layers": 2,
        "seq_len": input_len,
        "horizon": output_len,
        "log_samples": 1,
        "adj_gat": adjgat,
        "graphwave": gwv,
        "time_in_day_size": steps_per_day,
        "day_in_week_size": 7,
        "wave_type": "coif1",
        "wave_levels": 2,
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.STWave = STWave_ARGS


# ======== model: TCN =============== #
def TCN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"TCN_{ds_name}"
    MODEL.ARCH = TCN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "n_in":input_len, 
        "n_out":output_len, 
        "input_size":num_nodes, 
        "target_size":num_nodes, 
        "kernel_size":7, 
        "num_filters":3, 
        "num_layers":None, 
        "dilation_base":2, 
        "weight_norm":False, 
        "dropout":0.
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.TCN = TCN_ARGS


# ======== model: Transformer =============== #
def Transformer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Transformer_{ds_name}"
    MODEL.ARCH = Transformer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "n_in":input_len, 
        "n_out":output_len, 
        "input_size":num_nodes, 
        "target_size":num_nodes, 
        "d_model":64, 
        "nhead":4, 
        "num_encoder_layers":3, 
        "num_decoder_layers":3, 
        "dim_feedforward":512, 
        "dropout":0., 
        "activation":'relu', 
        "custom_encoder":None, 
        "custom_decoder":None
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.Transformer = Transformer_ARGS



# ======== model: Triformer =============== #
def Triformer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"Triformer_{ds_name}"
    MODEL.ARCH = Triformer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes": num_nodes,
        "lag": input_len,
        "horizon": output_len,
        "input_dim": 3,
        # default parameters described in the paper
        "channels": 32,
        "patch_sizes": [7, 4, 3, 2, 2] if input_len==336 else [4,3],
        "mem_dim": 5
        }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.Triformer = Triformer_ARGS


# ======== model: WaveNet =============== #
def WaveNet_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"WaveNet_{ds_name}"
    MODEL.ARCH = WaveNet
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "in_dim": 1,
        "out_dim": output_len,
        "residual_channels": 16,
        "dilation_channels": 16,
        "skip_channels": 64,
        "end_channels": 128,
        "kernel_size": 12,
        "blocks": 6,
        "layers": 3
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.WaveNet = WaveNet_ARGS