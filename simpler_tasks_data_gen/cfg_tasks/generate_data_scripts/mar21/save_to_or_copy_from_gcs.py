import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
from cfg_tasks.generate_data_scripts import concat_data, dedup_data, filter_data_by_length, split_data_train_val
from os import system as bash
import glob

def copy_from_gcs(path_to_get, save_dir):
  bash(f'mkdir -p {save_dir}')
  command = \
  f'gsutil -m cp -r \
    {path_to_get} \
    {save_dir}'
  print(command)
  bash(command)

def save_to_gcs(path_to_save, gcs_save_dir):
  command = \
  f'gsutil -m cp -r \
    {path_to_save} \
    {gcs_save_dir}'
  print(command)
  bash(command)


def get_vm_train_val_path_when_save_to_gcs(task):
  path_root = f'/mnt/disks/persist/Universal_LIME/cfg_tasks/generated_cfg_task_data/mar21/{task}/'
  return glob.glob(osp.join(path_root, 'train*.txt'))[0], glob.glob(osp.join(path_root, 'val*.txt'))[0]
  

def get_gcs_data_dir_when_save_to_gcs(task):
  return f'gs://n2formal-community-external-lime/universal_lime/generated_cfg_task_data/mar21/{task}/'


def get_gcs_data_dir_when_copy_from_gcs():
  return f'gs://n2formal-community-external-lime/universal_lime/generated_cfg_task_data/mar21/'

def get_vm_destination_dir_when_copy_from_gcs():
  return f'/mnt/disks/persist/Universal_LIME/cfg_tasks/generated_cfg_task_data/'

if sys.argv[1] == 'st':
  for t in ['induct', 'abduct', 'deduct']:
    train, val = get_vm_train_val_path_when_save_to_gcs(t)
    assert osp.isfile(train)
    assert osp.isfile(val)
    gcs_dest = get_gcs_data_dir_when_save_to_gcs(t)
    save_to_gcs(train, gcs_dest)
    save_to_gcs(val, gcs_dest)

elif sys.argv[1] == 'cf':
  gcs_data = get_gcs_data_dir_when_copy_from_gcs()
  vm_dest = get_vm_destination_dir_when_copy_from_gcs()
  copy_from_gcs(gcs_data, vm_dest)
else:
  raise NotImplementedError()
