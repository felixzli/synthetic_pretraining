import glob
import os

files_to_split_train_val = glob.glob('./data/11-29/*/deduped*')
train_count = 1_000_000
valid_count = 4_000
for path in files_to_split_train_val:
  os.system(f'python data_scripts/split_data_train_val.py --data_path {path} --train_count {train_count} --valid_count {valid_count}')