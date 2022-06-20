# final_exps/language_tasks/cnndm/runs/1final_init_exps/set/init_table_2/across_layer_grouping___init_scale.sh


import sys
import os.path as osp
from os import system as bash



file_to_copy = 'final_exps/language_tasks/cnndm/runs/1final_init_exps/set/init_table_2/across_layer_grouping___init_scale.sh'
dest_dir = 'final_exps/language_tasks/cnndm/runs/1final_init_exps/set/init_table_3'
init_ids = [
'preattnln_per_param_grouping___init_scale',
'per_param_grouping_exclude_preattnln___init_mean_std',

'relpos_per_param_grouping___init_scale',
'per_param_grouping_exclude_relpos___init_mean_std',

'qkvo_per_param_grouping___init_scale',
'per_param_grouping_exclude_qkvo___init_mean_std',

'mlp_per_param_grouping___init_scale',
'per_param_grouping_exclude_mlp___init_mean_std',

'premlpln_per_param_grouping___init_scale',
'per_param_grouping_exclude_premlpln___init_mean_std',
]

bash(f'mkdir -p {dest_dir}')
new_files = [f'{osp.join(dest_dir, init_id)}.sh' for init_id in init_ids]

for new_file in new_files:
  bash(f'cp {file_to_copy} {new_file}')


# for init