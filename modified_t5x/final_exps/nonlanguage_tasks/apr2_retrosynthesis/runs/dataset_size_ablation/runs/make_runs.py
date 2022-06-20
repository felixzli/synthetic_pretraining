from os import system as bash
import os.path as osp
import glob
import os
print(osp.dirname(__file__))
print(osp.dirname(osp.dirname(__file__)))
base_dir = osp.dirname(osp.dirname(__file__))
folders = glob.glob(f'{base_dir}/*/')
folders.remove(f'{osp.dirname(__file__)}/')
print(folders)
from os.path import dirname as dirname


def get_path_base(path):
  if path[-1] == '/':
    return path.split('/')[-2]
  else:
    return path.split('/')[-1]


for folder in folders:
  folder_base = get_path_base(folder)
  print(folder_base)
  folder_runs_path = osp.join(base_dir, 'runs',f'run_{folder_base}.sh')
  print(folder_runs_path)
  files_in_folder = glob.glob(f'{folder}/*')
  print(files_in_folder)
  with open(folder_runs_path, mode='w') as f:
    for file in files_in_folder:
      f.write(f'bash {file} $1\n')

  # bash(f'mkdir -p {base_dir+folder_base}')


