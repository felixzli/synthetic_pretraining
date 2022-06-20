import os
from os import system as bash
import os.path as osp

from setuptools import Command


for i in range(1,5):
  command =f'bash data_scripts/save_data_to_gcs.sh std_exps_1M_set_run{i}'
  bash(command)


for i in range(1,5):
  command =f'bash data_scripts/save_data_to_gcs.sh std_exps_1M_lime_run{i}'
  bash(command)



for i in range(1,5):
  assert osp.isfile(f'data_scripts/std_exps_1M_lime_run{i}/get_data.sh')
  command =f'bash data_scripts/std_exps_1M_lime_run{i}/get_data.sh'
  bash(command)


for i in range(1,5):
  assert osp.isfile(f'data_scripts/std_exps_1M_set_run{i}/get_data.sh')
  command =f'bash data_scripts/std_exps_1M_set_run{i}/get_data.sh'
  bash(command)

