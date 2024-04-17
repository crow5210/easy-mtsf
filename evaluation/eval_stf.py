import torch
import numpy as np
import os,sys
sys.path.append(os.path.abspath(__file__ + "/../.."))
from runner.runner import train_test
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="evaluation for spatial temeral forecasting")
    parser.add_argument('--epochs',         default=10, type=int,   help='train epochs')
    parser.add_argument('--val_interval',   default=1,  type=int,   help='validation interval')
    parser.add_argument('--seed',           default=1,  type=int,   help='random seed')
    parser.add_argument('--out_dir',        default='./checkpoints',   type=str,   help='result save directory')
    return parser.parse_args()


algos = ["STGCN","DCRNN","GraphWaveNet","DGCRN","D2STGNN","AGCRN","MTGNN","StemGNN","BGSLF","STDCN", "STNorm", "STID","STGODE","STWave",
        # "GTS","MegaCRN"
        ]
datas = ["PEMS-BAY","PEMS08","PEMS04","PEMS03","METR-LA"]

algos = ["STID"]
datas = ["PEMS04"]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args = parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

train_test(datas,algos,args.epochs,args.val_interval,args.out_dir,device)