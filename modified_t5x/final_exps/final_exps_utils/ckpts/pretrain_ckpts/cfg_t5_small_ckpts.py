import os.path as osp
from os import system as bash
from path_utils import parent_dir
import sys
import time


PRETRAIN_IDS = ['cfg_lime']
GCS_CKPT_BASE_DIR = osp.join('gs://n2formal-community-external-lime/universal_lime/t5x_checkpoints/', osp.basename(__file__)[:-3])
LOCAL_CKPT_BASE_DIR = osp.join('/mnt/disks/persist/t5_training_models/CHECKPOINTS_FROM_GCS/', osp.basename(__file__)[:-3])


def get_local_ckpt_path(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  return osp.join(LOCAL_CKPT_BASE_DIR, pretrain_id, 'checkpoint/')


def get_gcs_ckpt_path(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  return osp.join(GCS_CKPT_BASE_DIR, pretrain_id, 'checkpoint/')


def copy_ckpt_to_gcs(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  if pretrain_id == 'cfg_lime':
    # VM ?
    __pretrain_exp_ckpt_path = '/mnt/disks/persist/t5_training_models/final_exps/mar21_cfg/mix3_pretrain_t5small/checkpoint_524288'
  else:
    raise NotADirectoryError
  pretrain_exp_ckpt_path = osp.join(parent_dir(__pretrain_exp_ckpt_path), 'checkpoint/')
  bash(f'rm -rf {pretrain_exp_ckpt_path}')
  assert len(pretrain_exp_ckpt_path) < len(__pretrain_exp_ckpt_path)
  bash(f'cp -r {__pretrain_exp_ckpt_path} {pretrain_exp_ckpt_path}')
  gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)
  bash(f'gsutil -m rm -rf {parent_dir(gcs_ckpt_path)}')
  print(pretrain_exp_ckpt_path)
  print(gcs_ckpt_path)
  command = f'gsutil -m cp -r {pretrain_exp_ckpt_path} {parent_dir(gcs_ckpt_path)}/'

  print(command)
  bash(command)
  print('- finished copy_ckpt_to_gcs -')


def copy_ckpt_from_gcs_to_local(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  local_ckpt_path = get_local_ckpt_path(pretrain_id)
  gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)

  bash(f'rm -rf {parent_dir(local_ckpt_path)}')
  bash(f'mkdir -p {parent_dir(local_ckpt_path)}')
  command = f'gsutil -m cp -r {gcs_ckpt_path} {parent_dir(local_ckpt_path)}'

  print(command)
  bash(command)
  print('- finished copy_ckpt_from_gcs_to_local -')



if __name__ == '__main__':
  pretrain_id = sys.argv[1]
  if len(sys.argv) > 2 and sys.argv[2] == 'copy_ckpt_to_gcs':
    copy_ckpt_to_gcs(pretrain_id)
    time.sleep(0.5)
  copy_ckpt_from_gcs_to_local(pretrain_id)
  time.sleep(0.5)
  print(get_local_ckpt_path(pretrain_id))
  bash(f'ls {get_local_ckpt_path(pretrain_id)}')
  print(get_gcs_ckpt_path(pretrain_id))
  print(get_local_ckpt_path(pretrain_id))

  
