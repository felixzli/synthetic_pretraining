import os.path as osp
from os import system as bash
from path_utils import parent_dir
import sys


def copy_original_ckpt_to_gcs(original_ckpt_path, gcs_ckpt_path):
  ckpt_path = osp.join(parent_dir(original_ckpt_path), 'checkpoint/')
  bash(f'rm -rf {ckpt_path}')
  assert len(ckpt_path) < len(original_ckpt_path)
  bash(f'cp -r {original_ckpt_path} {ckpt_path}')
  # gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)
  bash(f'gsutil -m rm -rf {parent_dir(gcs_ckpt_path)}')
  print(ckpt_path)
  print(gcs_ckpt_path)
  command = f'gsutil -m cp -r {ckpt_path} {parent_dir(gcs_ckpt_path)}/'

  print(command)
  bash(command)
  print('- finished copy_ckpt_to_gcs -')

