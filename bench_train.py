import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from timegan import timegan

from bench_utils import DeclareArg

# DATA_PATH = './data/ETT-small/ETTh1.csv'
# SEQ_LEN = 24
# ITERATIONS = 10
# BATCH_SIZE = 128
# HIDDEN_DIM = 24
# NUM_LAYERS = 3
# MODULE = 'gru'
DATA_PATH = DeclareArg('data_path', str, './data/ETT-small/ETTh1.csv', 'Path to the data file')
SEQ_LEN = DeclareArg('seq_len', int, 24, 'Sequence length for time series data')
ITERATIONS = DeclareArg('iterations', int, 10, 'Number of training iterations for TimeGAN')
BATCH_SIZE = DeclareArg('batch_size', int, 128, 'Batch size for training TimeGAN')
HIDDEN_DIM = DeclareArg('hidden_dim', int, 24, 'Hidden dimension size for TimeGAN')
NUM_LAYERS = DeclareArg('num_layers', int, 3, 'Number of layers for TimeGAN')
MODULE = DeclareArg('module', str, 'gru', 'RNN module type for TimeGAN (gru, lstm, or lstmLN)')


def load_data(path, seq_len):
    df = pd.read_csv(path)
    data = df.iloc[:, 1:].values

    # Normalizing the data
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data = (data - data_min) / (data_max - data_min + 1e-7)

    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return sequences


ori_data = load_data(DATA_PATH, SEQ_LEN)

parameters = {
    'module': MODULE,
    'hidden_dim': HIDDEN_DIM,
    'num_layer': NUM_LAYERS,
    'iterations': ITERATIONS,
    'batch_size': BATCH_SIZE
}

generated_data = timegan(ori_data, parameters)