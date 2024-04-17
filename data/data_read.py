import pandas as pd
import numpy as np

def read_csv(data_file_path,target_channel):
    df = pd.read_csv(data_file_path)
    try:
        df_index = pd.to_datetime(df["date"].values, format="%Y-%m-%d %H:%M")
    except:
        try:
            df_index = pd.to_datetime(df["date"].values, format="%Y/%m/%d %H:%M")
        except:
            df_index = pd.to_datetime(df["date"].values, format="%Y-%m-%d %H:%M:%S")
    df = df[df.columns[1:]]
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    return data,df_index

def read_h5(data_file_path,target_channel):
    df = pd.read_hdf(data_file_path)
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    return data,df.index

def read_npz(data_file_path,target_channel):
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    return data,None

def read_xlsx(data_file_path,target_channel):
    df = pd.read_excel(data_file_path)
    data = df.values
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    return data,None
