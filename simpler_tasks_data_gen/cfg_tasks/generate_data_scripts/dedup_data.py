import sys
import glob
import os
import os.path as osp
import pandas as pd


def get_dedup_path(path_data_to_dedup):
  data_dir = osp.dirname(path_data_to_dedup)
  dedup_data_path = osp.join(data_dir, 'dedup_' + osp.basename(path_data_to_dedup))
  return dedup_data_path


def deduplicate_generated_data(data_path):
  dedup_data_path = get_dedup_path(data_path)
  data_inputs_set = set()

  num_dedup_data = 0

  with open(data_path, 'r') as data_file, open(dedup_data_path, 'w') as dedup_data_file:
    for line in data_file:
      input = line.split('\t')[0]
      if input in data_inputs_set:
        continue
      data_inputs_set.add(input)
      num_dedup_data += 1
      dedup_data_file.write(line)
  print('-'*10)
  print(f'DEDUP DATA PATH: {dedup_data_path}')
  print(f'\tNUM DEDUP DATA: {num_dedup_data}')
  os.system(f'wc -l {dedup_data_path}')
  print('-'*10)

if __name__ == '__main__':
  deduplicate_generated_data(sys.argv[1])
