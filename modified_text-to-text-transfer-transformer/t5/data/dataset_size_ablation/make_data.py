import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
import seqio
from t5.data.tasks import TaskRegistry
from build_custom_datasets.utils import head_files
import numpy as np
# task = sys.argv[1]
from os import system as bash


def get_dataset_iterator(mixture_or_task_naame, split, amount=9999999999):
  task_registry_ds = TaskRegistry.get_dataset(mixture_or_task_naame, sequence_length={"inputs": 2048, "targets": 2048},
                                                    split=split, shuffle=True)
  return task_registry_ds.take(amount)


def make_data(mixture_or_task_name, split, amount, save_path, debug=False):
  dataset_iter = get_dataset_iterator(mixture_or_task_name, split, amount)
  bash(f'mkdir -p {osp.dirname(save_path)}')
  with open(save_path, mode='w') as f:
    for i, data in enumerate(dataset_iter):
      if i % 10000 == 0:
        print(f'{i} data processed')
      if 'inputs_pretokenized' not in data.keys():
        raise NotImplementedError

      inp_pre = data['inputs_pretokenized'].numpy().decode('utf-8')
      tgt_pre = data['targets_pretokenized'].numpy().decode('utf-8')
      if '\n' in tgt_pre:
        tgt_pre = tgt_pre.replace('\n', ' ')
      if debug: 
        print(data.keys())
        print('input')
        print(inp_pre)
        print('\n\ntgt')
        print(tgt_pre)
        # breakpoint()
        if i == 1:
          break
      
      f.write(f'{inp_pre}\t{tgt_pre}\n')

      
    

def make_train_data(mixture_or_task_name, amount, save_dir='t5/data/dataset_size_ablation/data'):
  save_path = osp.join(save_dir, mixture_or_task_name, f'{amount}_train.txt')
  make_data(mixture_or_task_name, 'train', amount, save_path)


def make_valid_data(mixture_or_task_name, save_dir='t5/data/dataset_size_ablation/data'):
  save_path = osp.join(save_dir, mixture_or_task_name, 'valid.txt')
  make_data(mixture_or_task_name, 'validation', 99999999, save_path)


def make_test_data(mixture_or_task_name, save_dir='t5/data/dataset_size_ablation/data'):
  save_path = osp.join(save_dir, mixture_or_task_name, 'test.txt')
  make_data(mixture_or_task_name, 'test', 99999999, save_path)


def check_shuffle(mixture_or_task_name):
  print('take 1...')
  make_data(mixture_or_task_name, 'train', 1,'foo', debug = True)
  print('take 1...')
  make_data(mixture_or_task_name, 'train', 1, 'foo', debug = True)

#"cnn_dailymail_v002"
if __name__=='__main__':
  # check_shuffle('squad_v010_allanswers')

  # tasks=['squad_v010_allanswers']
  # for t in tasks:
    # make_valid_data(t)
    # make_test_data(t)
    # make_train_data(t, 1000)
    # make_train_data(t, 10000)
    # make_train_data(t, 100000)
  #   bash(f'wc -l t5/data/dataset_size_ablation/data/{t}/*.txt')

  # tasks=["cnn_dailymail_v002"]
  # for t in tasks:
  #   check_shuffle(t)

  #   make_valid_data(t)
  #   make_test_data(t)
  #   make_train_data(t, 1000)
  #   # make_train_data(t, 10000)
  #   make_train_data(t, 100000)
  #   bash(f'wc -l t5/data/dataset_size_ablation/data/{t}/*.txt')


    t = 'single_step_retrosynthesis'

    make_valid_data(t)
    make_test_data(t)
    make_train_data(t, 1000)
    make_train_data(t, 10000)
    make_train_data(t, 100000)
    bash(f'wc -l t5/data/dataset_size_ablation/data/{t}/*.txt')


    # 87599 t5/data/dataset_size_ablation/data/squad_v010_allanswers/100000_train.txt
    # 10000 t5/data/dataset_size_ablation/data/squad_v010_allanswers/10000_train.txt
    #  1000 t5/data/dataset_size_ablation/data/squad_v010_allanswers/1000_train.txt
    # 87599 t5/data/dataset_size_ablation/data/squad_v010_allanswers/test.txt
    # 10570 t5/data/dataset_size_ablation/data/squad_v010_allanswers/valid.txt

    # 100000 t5/data/dataset_size_ablation/data/cnn_dailymail_v002/100000_train.txt
    # 1000 t5/data/dataset_size_ablation/data/cnn_dailymail_v002/1000_train.txt
    # 11490 t5/data/dataset_size_ablation/data/cnn_dailymail_v002/test.txt
    # 13368 t5/data/dataset_size_ablation/data/cnn_dailymail_v002/valid.txt



