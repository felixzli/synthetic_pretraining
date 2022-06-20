import glob
import os
import sys

args = sys.argv
# files_to_split_train_val = glob.glob('./data/1-4/*/deduped*')
train_count = int(args[1])
valid_count = int(args[2])
files_to_split_train_val = glob.glob(args[3])
for path in files_to_split_train_val:
  os.system(f'python data_scripts/split_data_train_val.py --data_path {path} --train_count {train_count} --valid_count {valid_count}')