import pathlib

#import finrl

import pandas as pd
import datetime
import os
#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10


#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#DATASET_DIR = PACKAGE_ROOT / "data"

# data
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = "C:/Users/user/Desktop/DRL/source/data/dow_30_2009_2020_v2.csv"
TRAINING_DATA_FILE2 = "C:/Users/user/Desktop/DRL/source/data/done_data_amen5.csv"

now = datetime.datetime.now()
#TRAINED_MODEL_DIR = f"trained_models/{now}"
TRAINED_MODEL_DIR = f"C:/Users/user/Desktop/DRL/source/trained_models/result"
#os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "C:/Users/user/Desktop/DRL/source/data/turbulence_index.csv"

FEATURES_ADDED = 'C:/Users/user/Desktop/DRL/feature_fama.csv'

TESTING_DATA_FILE = "test.csv"


