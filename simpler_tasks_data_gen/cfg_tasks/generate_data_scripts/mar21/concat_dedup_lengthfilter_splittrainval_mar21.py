import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
from cfg_tasks.generate_data_scripts import concat_data, dedup_data, filter_data_by_length, split_data_train_val
from os import system as bash

# taskss = ['deduct', 'induct', 'abduct']
taskss = ['abduct']

for task in taskss:
  # concat
  concat_path = f'cfg_tasks/generated_cfg_task_data/mar21/{task}/all_processes_1100000_data_concat_{task}.txt'
  assert osp.isdir(osp.dirname(concat_path))
  command = f'python cfg_tasks/generate_data_scripts/concat_data.py \
  cfg_tasks/generated_cfg_task_data/mar21/{task}/\*process\*generated\*.txt {concat_path}'
  print(command)
  bash(command)

  # dedup
  assert osp.isfile(concat_path)
  command = f'python cfg_tasks/generate_data_scripts/dedup_data.py {concat_path}'
  bash(command)


  # length filter
  dedup_path = dedup_data.get_dedup_path(concat_path)
  print('-'*20)
  print(dedup_path)
  max_src_len = 1024
  max_tgt_len = 1024
  bash(f'python cfg_tasks/generate_data_scripts/filter_data_by_length.py {dedup_path} {max_src_len} {max_tgt_len}')
  

  # split train val
  length_filter_path = filter_data_by_length.get_new_data_path(dedup_path, max_src_len, max_tgt_len)
  print('-'*20)
  print(length_filter_path)
  train_count = 1000000
  val_count = 10000
  bash(f'python cfg_tasks/generate_data_scripts/split_data_train_val.py --data_path {length_filter_path} --train_count {train_count} --valid_count {val_count}')

  train_path, val_path = split_data_train_val.get_train_valid_paths(length_filter_path, train_count, val_count)
  bash(f'wc -l {train_path} {val_path}')


