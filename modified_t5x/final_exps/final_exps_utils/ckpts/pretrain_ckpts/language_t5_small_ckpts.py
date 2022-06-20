import os.path as osp
from os import system as bash
import sys
from path_utils import parent_dir


PRETRAIN_IDS = ['offshelf']


def get_ckpt_name_given_pretrain_id(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  return f'{pretrain_id}_t5_small'


def get_local_ckpt_path(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  pretrain_name = get_ckpt_name_given_pretrain_id(pretrain_id)
  return f'/mnt/disks/persist/t5_training_models/CHECKPOINTS_FROM_GCS/{pretrain_name}/checkpoint/'


def get_gcs_ckpt_path(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  pretrain_name = get_ckpt_name_given_pretrain_id(pretrain_id)
  return f'gs://n2formal-community-external-lime/universal_lime/t5x_checkpoints/{pretrain_name}/checkpoint/'


def copy_ckpt_from_gcs_to_local(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  local_ckpt_path = get_local_ckpt_path(pretrain_id)
  gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)
  bash(f'mkdir -p {parent_dir(local_ckpt_path)}')
  command = f'gsutil -m cp -r {gcs_ckpt_path} {parent_dir(local_ckpt_path)}'
  bash(command)


if __name__ == '__main__':
  pretrain_id = sys.argv[1]
  # if len(sys.argv) > 2 and sys.argv[2] == 'copy_ckpt_to_gcs':
  #   copy_ckpt_to_gcs(pretrain_id)
  #   time.sleep(0.5)
  # copy_ckpt_from_gcs_to_local(pretrain_id)
  # time.sleep(0.5)
  print(get_local_ckpt_path(pretrain_id))
  bash(f'ls {get_local_ckpt_path(pretrain_id)}')
  print(get_local_ckpt_path(pretrain_id))
