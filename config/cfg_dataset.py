from easydict import EasyDict
from data.data_read import read_csv,read_h5,read_npz,read_xlsx


CFG_DATASET = EasyDict()
# ======== dataset: PEMS08 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "PEMS08"
DATASET_ARGS.NUM_NODES = 170
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_npz
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)

CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS


# ======== dataset: PEMS03 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "PEMS03"
DATASET_ARGS.NUM_NODES = 358
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_npz
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)

CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

# ======== dataset: PEMS04 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "PEMS04"
DATASET_ARGS.NUM_NODES = 307
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_npz
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)

CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS



# ======== dataset: PEMS-BAY =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "PEMS-BAY"
DATASET_ARGS.NUM_NODES = 325
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_h5
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.h5".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS


# ======== dataset: METR-LA =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "METR-LA"
DATASET_ARGS.NUM_NODES = 207
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_h5
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.h5".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

# ======== dataset: Weather =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "Weather"
DATASET_ARGS.NUM_NODES = 21
DATASET_ARGS.STEPS_PER_DAY = 144
DATASET_ARGS.READ_DATA_FUNC = read_csv
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS


# ======== dataset: ExchangeRate =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "ExchangeRate"
DATASET_ARGS.NUM_NODES = 8
DATASET_ARGS.STEPS_PER_DAY = 1
DATASET_ARGS.READ_DATA_FUNC = read_csv
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS


# ======== dataset: Electricity =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "Electricity"
DATASET_ARGS.NUM_NODES = 321
DATASET_ARGS.STEPS_PER_DAY = 24
DATASET_ARGS.READ_DATA_FUNC = read_csv
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS



# ======== dataset: ETTh1 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "ETTh1"
DATASET_ARGS.NUM_NODES = 7
DATASET_ARGS.STEPS_PER_DAY = 24
DATASET_ARGS.READ_DATA_FUNC = read_csv
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

# ======== dataset: ETTh2 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "ETTh2"
DATASET_ARGS.NUM_NODES = 7
DATASET_ARGS.STEPS_PER_DAY = 24
DATASET_ARGS.READ_DATA_FUNC = read_csv
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

# ======== dataset: ETTm1 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "ETTm1"
DATASET_ARGS.NUM_NODES = 7
DATASET_ARGS.STEPS_PER_DAY = 24 * 4
DATASET_ARGS.READ_DATA_FUNC = read_csv
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

# ======== dataset: ETTm2 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "ETTm2"
DATASET_ARGS.NUM_NODES = 7
DATASET_ARGS.STEPS_PER_DAY = 24 * 4
DATASET_ARGS.READ_DATA_FUNC = read_csv
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = True                  # if add day_of_month feature
DATASET_ARGS.DOY = True                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS



# ======== dataset: BeijingAirQuality =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "BeijingAirQuality"
DATASET_ARGS.NUM_NODES = 7
DATASET_ARGS.STEPS_PER_DAY = 24
DATASET_ARGS.READ_DATA_FUNC = read_xlsx
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.xlsx".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

