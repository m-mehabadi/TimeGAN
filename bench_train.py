import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import bench

from timegan import timegan
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics

from bench_utils import DeclareArg

DATA_PATH = DeclareArg('data_path', str, './data/ETT-small/ETTh1.csv', 'Path to the data file')
SEQ_LEN = DeclareArg('seq_len', int, 24, 'Sequence length for time series data')
ITERATIONS = DeclareArg('iterations', int, 10, 'Number of training iterations for TimeGAN')
BATCH_SIZE = DeclareArg('batch_size', int, 128, 'Batch size for training TimeGAN')
HIDDEN_DIM = DeclareArg('hidden_dim', int, 24, 'Hidden dimension size for TimeGAN')
NUM_LAYERS = DeclareArg('num_layers', int, 3, 'Number of layers for TimeGAN')
MODULE = DeclareArg('module', str, 'gru', 'RNN module type for TimeGAN (gru, lstm, or lstmLN)')
OUTPUT_DIR = DeclareArg('output_dir', str, './_bench_output', 'Output directory for results')
EXPERIMENT_NAME = DeclareArg('experiment_name', str, 'experiment', 'Experiment name')
RUN_NAME = DeclareArg('run_name', str, 'run_0', 'Run name')
SEED = DeclareArg('seed', int, 0, 'Random seed')


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

def main():
    
  bench.start_run()
  
  ori_data = load_data(DATA_PATH, SEQ_LEN)
  
  parameters = {
    'module': MODULE,
    'hidden_dim': HIDDEN_DIM,
    'num_layer': NUM_LAYERS,
    'iterations': ITERATIONS,
    'batch_size': BATCH_SIZE
  }
  
  # Main worker
  generated_data = timegan(ori_data, parameters)
  
  # Save generated data
  from pathlib import Path
  output_path = Path(bench.get_output_dir())
  output_path.mkdir(parents=True, exist_ok=True)
  np.save(str(output_path / "generated.npy"), np.asarray(generated_data))
  
  # disc_score = discriminative_score_metrics(ori_data, generated_data)
  # pred_score = predictive_score_metrics(ori_data, generated_data)
  
  bench.log_metrics({
    # "discriminative_score": disc_score,
    # "predictive_score": pred_score,
    "num_sequences": float(len(ori_data)),
    "seq_len": float(SEQ_LEN),
    "num_generated": float(len(generated_data))
  })
  
  bench.end_run(status="FINISHED")

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print("Error: {}".format(str(e)))
    bench.end_run(status="FAILED")