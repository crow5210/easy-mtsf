from easydict import EasyDict
from utils.metrics import masked_mae, masked_mape, masked_wape, dtw


# ================= common ================= #
CFG_GENERAL = EasyDict()
CFG_GENERAL.METRICS = {"MAE": masked_mae, 
                       "MAPE": masked_mape, 
                       "WAPE": masked_wape, 
                       "DTW":dtw
                       }
CFG_GENERAL.NULL_VAL = 0.0

CFG_GENERAL.OPTIM = EasyDict()
CFG_GENERAL.OPTIM.TYPE = "Adam"
CFG_GENERAL.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}

CFG_GENERAL.LR_SCHEDULER = EasyDict()
CFG_GENERAL.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG_GENERAL.LR_SCHEDULER.PARAM= {
    "milestones":[20, 40, 60, 80],
    "gamma":0.5
}

# ================= dataset ================= #
CFG_GENERAL.DATASET = EasyDict()
CFG_GENERAL.DATASET.HISTORY_SEQ_LEN = 336
CFG_GENERAL.DATASET.FUTURE_SEQ_LEN = 336
CFG_GENERAL.DATASET.TRAIN_RATIO = 0.6
CFG_GENERAL.DATASET.VALID_RATIO = 0.2
CFG_GENERAL.DATASET.NORM_EACH_CHANNEL = False
# ================= train ================= #
CFG_GENERAL.TRAIN = EasyDict()
CFG_GENERAL.TRAIN.BATCH_SIZE = 32
CFG_GENERAL.TRAIN.PREFETCH = False
CFG_GENERAL.TRAIN.SHUFFLE = True
CFG_GENERAL.TRAIN.NUM_WORKERS = 0
CFG_GENERAL.TRAIN.PIN_MEMORY = False

# ================= validate ================= #
CFG_GENERAL.VAL = EasyDict()
CFG_GENERAL.VAL.BATCH_SIZE = 32
CFG_GENERAL.VAL.PREFETCH = False
CFG_GENERAL.VAL.SHUFFLE = False
CFG_GENERAL.VAL.NUM_WORKERS = 0
CFG_GENERAL.VAL.PIN_MEMORY = False

# ================= test ================= #
CFG_GENERAL.TEST = EasyDict()
CFG_GENERAL.TEST.BATCH_SIZE = 32
CFG_GENERAL.TEST.PREFETCH = False
CFG_GENERAL.TEST.SHUFFLE = False
CFG_GENERAL.TEST.NUM_WORKERS = 0
CFG_GENERAL.TEST.PIN_MEMORY = False
# CFG_GENERAL.TEST.HORIZON = [96,192,336]