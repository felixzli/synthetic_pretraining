import os.path as osp
from os import system as bash
from path_utils import parent_dir
import sys


def copy_ckpt_from_gcs_to_local(gcs_ckpt_path, local_ckpt_path):
  bash(f'rm -rf {parent_dir(local_ckpt_path)}')
  bash(f'mkdir -p {parent_dir(local_ckpt_path)}')
  command = f'gsutil -m cp -r {gcs_ckpt_path} {parent_dir(local_ckpt_path)}'

  print(command)
  bash(command)
  print('- finished copy_ckpt_from_gcs_to_local -')