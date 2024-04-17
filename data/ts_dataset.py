
from torch.utils.data import Dataset
from utils.serialization import load_pkl
import torch

class TsDataset(Dataset):
    def __init__(self, data_file_path: str, index_file_path: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        self.index = load_pkl(index_file_path)[mode]

    def __getitem__(self, index: int) -> tuple:

        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]

        return future_data, history_data

    def __len__(self):
        return len(self.index)