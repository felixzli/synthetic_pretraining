import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from os import system as bash
from path_utils import make_path_with_prefix_added_to_basename


def src_len_function(src):
  src = eval(src)
  assert type(src) == list
  return len(src)


def get_new_data_path(data_path, src_max_len, tgt_max_len):
  return make_path_with_prefix_added_to_basename(data_path, f'src{src_max_len}_tgt{tgt_max_len}_')


if __name__ == '__main__':
  tgt_len_function = src_len_function

  data_path = sys.argv[1]


  src_lengths = []
  tgt_lengths = []
  with open(data_path, mode='r') as data_file:
    for i, line in enumerate(data_file):
      try:
        src, tgt = line.split('\t')
        if tgt[-1] == '\n':
          tgt = tgt[:-1]
      except:
        print(i)
        print(line)
        exit()
      # if len_function(src) < src_max_length and len_function(tgt) < tgt_max_length:

      src_lengths.append(src_len_function(src))
      tgt_lengths.append(tgt_len_function(tgt))
  src_lengths = pd.Series(src_lengths)
  tgt_lengths = pd.Series(tgt_lengths)


  percentiles = [0.25,0.5,0.75,0.8,0.85,0.9,0.95,0.98]
  print(f'src lengths =======')
  print(src_lengths.describe(percentiles=percentiles))
  print('tgt lengths =======')
  print(tgt_lengths.describe(percentiles=percentiles))

  bash(f'wc -l {data_path}')
