import os
import os.path as osp
import glob

base_dir = './data/1-4'
all_task_train_paths = base_dir + '/*/train*'
all_task_valid_paths = base_dir + '/*/valid*'

concat_data_dir = base_dir + '/all_tasks'
concat_train_path = concat_data_dir + '/train.txt'
concat_valid_path = concat_data_dir + '/valid.txt'

os.system(f'mkdir -p {concat_data_dir}')
os.system(f'cat {all_task_train_paths} > {concat_train_path}')
os.system(f'cat {all_task_valid_paths} > {concat_valid_path}')

os.system(f'wc -l {concat_data_dir}/*')