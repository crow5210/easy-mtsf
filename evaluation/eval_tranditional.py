import os,sys
sys.path.append(os.path.abspath(__file__ + "/../.."))
from config.cfg_algorithm import CFG_MODEL
import argparse
from data.data_loader import build_dataset_darts
from utils.plot import plot_multi_result_darts

def parse_args():
    parser = argparse.ArgumentParser(description="easy-mtsf.")
    parser.add_argument('--test_ratio', default=0.2, type=float,   help='train epochs')
    parser.add_argument('--horizen',    default=12,  type=int,     help='train epochs')
    parser.add_argument('--out_dir',        default='./checkpoints/tranditional',   type=str,   help='result save directory')
    return parser.parse_args()


def train(datas,algos,test_ratio,out_dir):
    for data in datas:
        for algo in algos:  
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            datasets = build_dataset_darts(data,test_ratio)
            model_cfg = CFG_MODEL[algo](data,1,None,None,None,None)
            model = model_cfg["ARCH"](**model_cfg["PARAM"])
            model.fit(datasets["train"])
            model.save(os.path.join(out_dir, f"{algo}_{data}.pkl"))


def test(datas,algos,test_ratio,horizen,out_dir):
    for data in datas:
        for algo in algos:
            datasets = build_dataset_darts(data,test_ratio)
            model_cfg = CFG_MODEL[algo](data,1,None,None,None,None)
            model = model_cfg["ARCH"](**model_cfg["PARAM"])
            model = model.load(os.path.join(out_dir, f"{algo}_{data}.pkl"))
            preds = model.predict(datasets["test"],horizen)
            plot_multi_result_darts(datasets["test"],preds,f"{algo}_{data}",save_dir=out_dir)


if __name__ == "__main__":
    algos_simple = ["NAIVE","ES","Theta","ARIMA","Kalman"]
    algos_ML = ["LGBM","Random_Forest"]
    
    datas = ["air","icecream"]
    algos = algos_simple + algos_ML
    args = parse_args()

    train(datas,algos,args.test_ratio,args.out_dir)
    test(datas,algos,args.test_ratio,args.horizen,args.out_dir)
