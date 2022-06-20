import sys
import glob
import os
import os.path as osp


def concat_data(paths_to_concat, concat_path):
  print('-'*20)
  print(f'concatenating {len(paths_to_concat)}')
  print(f'FIRST 5 PATHS TO CONCAT: {paths_to_concat[:5]}')

  print(f'CONCAT PATH: {concat_path}')
  paths_to_concat = ' '.join(paths_to_concat)
  os.system(f'cat {paths_to_concat} > {concat_path}')


if __name__ == '__main__':
  paths_to_concat = glob.glob(sys.argv[1])
  concat_path = sys.argv[2]
  concat_data(paths_to_concat, concat_path)
  os.system(f'wc -l {concat_path}')

# python cfg_tasks/generate_data_scripts/concat_data.py cfg_tasks/generated_cfg_task_data/mar20/deduct/process*.txt all_processes_1100000_data_concat_deduct.txt