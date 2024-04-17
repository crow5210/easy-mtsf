import os
from data.transform import standard_transform
import numpy as np
import shutil
import pickle
from data.ts_dataset import TsDataset
from torch.utils.data import DataLoader
from utils.serialization import load_pkl
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MaxAbsScaler

def generate_data(args):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """
    if not os.path.exists(args["OUTPUT_DIR"]):
        os.makedirs(args["OUTPUT_DIR"])

    target_channel = args.get("TARGET_CHANNEL",[0])
    future_seq_len = args["FUTURE_SEQ_LEN"]
    history_seq_len = args["HISTORY_SEQ_LEN"]
    output_dir = args["OUTPUT_DIR"]
    train_ratio = args["TRAIN_RATIO"]
    valid_ratio = args["VALID_RATIO"]
    data_file_path = args["DATA_FILE_PATH"]
    norm_each_channel = args["NORM_EACH_CHANNEL"]
    graph_file_path = args.get("GRAPH_FILE_PATH",None)
    steps_per_day = args.get("STEPS_PER_DAY",None)
    add_time_of_day = args.get("TOD",True)
    add_day_of_week = args.get("DOW",True)
    add_day_of_month = args.get("DOM",False)
    add_day_of_year = args.get("DOY",False)
    
    if_rescale = not norm_each_channel 

    # read data
    data,df_index = args["READ_DATA_FUNC"](data_file_path,target_channel)
    print("raw time series shape: {0}".format(data.shape))

    # split data
    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num + valid_num : train_num + valid_num + test_num]

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        if steps_per_day is None:
            tod = (df_index.values - df_index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        else:
            tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
            tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        if steps_per_day is None:
            dow = df_index.dayofweek / 7
        else:
            dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
            dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if add_day_of_month:
        # numerical day_of_month
        dom = (df_index.day - 1 ) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

    if add_day_of_year:
        # numerical day_of_year
        doy = (df_index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)

    # copy adj
    if graph_file_path is not None and os.path.exists(graph_file_path):
        shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")

def check_file(data_file_path: str, index_file_path: str, scaler_file_path: str):
    if not os.path.isfile(data_file_path):
        raise FileNotFoundError("can not find data file ")
    if not os.path.isfile(index_file_path):
        raise FileNotFoundError("can not find index file ")
    if not os.path.isfile(scaler_file_path):
        raise FileNotFoundError("can not find scaler file ")
    

def build_dataset(cfg_general,cfg_dataset):
    datasets={}
    dataset_dir = cfg_dataset.OUTPUT_DIR
    input_len = cfg_general.DATASET.HISTORY_SEQ_LEN
    output_len = cfg_general.DATASET.FUTURE_SEQ_LEN
    rescale = not cfg_general.DATASET.NORM_EACH_CHANNEL

    data_file_path = f'{dataset_dir}/data_in_{input_len}_out_{output_len}_rescale_{rescale}.pkl'
    index_file_path = f'{dataset_dir}/index_in_{input_len}_out_{output_len}_rescale_{rescale}.pkl'
    scaler_file_path = f'{dataset_dir}/scaler_in_{input_len}_out_{output_len}_rescale_{rescale}.pkl'

    try:
        check_file(data_file_path,index_file_path,scaler_file_path)
    except:
        ds_args = cfg_dataset.copy()
        ds_args.update(cfg_general.DATASET)
        generate_data(ds_args)

    ds_train = TsDataset(data_file_path,index_file_path,"train")
    datasets["train"] = DataLoader(ds_train,
                                    collate_fn  =cfg_general.TRAIN.get('COLLATE_FN', None),
                                    batch_size  =cfg_general.TRAIN.get('BATCH_SIZE', 1),
                                    shuffle     =cfg_general.TRAIN.get('SHUFFLE', False),
                                    num_workers =cfg_general.TRAIN.get('NUM_WORKERS', 0),
                                    pin_memory  =cfg_general.TRAIN.get('PIN_MEMORY', False))

    ds_val = TsDataset(data_file_path,index_file_path,"valid")
    datasets["val"] = DataLoader(ds_val,
                                    collate_fn  =cfg_general.VAL.get('COLLATE_FN', None),
                                    batch_size  =cfg_general.VAL.get('BATCH_SIZE', 1),
                                    shuffle     =cfg_general.VAL.get('SHUFFLE', False),
                                    num_workers =cfg_general.VAL.get('NUM_WORKERS', 0),
                                    pin_memory  =cfg_general.VAL.get('PIN_MEMORY', False))

    ds_test = TsDataset(data_file_path,index_file_path,"test")
    datasets["test"] = DataLoader(ds_test,
                                    collate_fn  =cfg_general.TEST.get('COLLATE_FN', None),
                                    batch_size  =cfg_general.TEST.get('BATCH_SIZE', 1),
                                    shuffle     =cfg_general.TEST.get('SHUFFLE', False),
                                    num_workers =cfg_general.TEST.get('NUM_WORKERS', 0),
                                    pin_memory  =cfg_general.TEST.get('PIN_MEMORY', False))
    
    datasets["name"] = cfg_dataset.NAME
    datasets["scaler"] = load_pkl(scaler_file_path)
    return datasets


def build_dataset_darts(dataSet,test_ratio):
    datafile = f"datasets/raw_data/{dataSet}_{test_ratio}.pickle"
    if os.path.isfile(datafile):
        with open(datafile, "rb") as f:
            datasets = pickle.load(f)
    else:
        print(f"building {dataSet} TimeSeries...")
        series = []
        # Read DataFrame
        if dataSet=="air":
            data = pd.read_csv("datasets/raw_data/AirPassengers/AirPassengers.csv",index_col=0).values
        elif dataSet=="icecream":
            data = pd.read_csv("datasets/raw_data/ice_cream/ice_cream_heater.csv",index_col=0).values
        elif dataSet=="Electricity":
            data = pd.read_csv("datasets/raw_data/Electricity/Electricity.csv",index_col=0).values
        else:
            print("dataset name must be one of:[air,sunsport,icecream,m4,elec]")
            return
        
        for i in range(data.shape[1]):
            series.append(TimeSeries.from_values(data[:,i]))

        # Split train/test
        print("splitting train/test...")
        val_len = round(data.shape[0]*test_ratio)
        train = [s[:-val_len] for s in series]
        test = [s[-val_len:] for s in series]

        # Scale so that the largest value is 1
        print("scaling...")
        scaler = Scaler(scaler=MaxAbsScaler())
        datasets = {}
        datasets["train"] = scaler.fit_transform(train)
        datasets["test"] = scaler.transform(test)
        datasets["scaler"] = scaler

        print(
            "done. There are {} series, with average training length {}".format(
                len(train), np.mean([len(s) for s in train])
            )
        )
        with open(datafile, 'wb') as f:
            pickle.dump(datasets, f)
    return datasets