import sys
import os.path as osp
from os import system as bash



file_to_copy = 'final_exps/language_tasks/cnndm/runs/1final_init_exps/LIME/test.sh'
dest_dir = 'final_exps/language_tasks/cnndm/runs/1final_init_exps/LIME'
init_ids = [
'per_param_grouping___init_mean_std',
'layer_params_per_param_grouping___init_scale',
'layer_params_across_layer_grouping___init_scale',

'layer_params_across_layer_big_grouping___init_scale',
'layer_params_across_layer_big_big_grouping___init_scale',

'attention_across_layer_grouping___init_scale',
'mlp_across_layer_grouping___init_scale',
'premlpln_across_layer_grouping___init_scale',
'preattnln_across_layer_grouping___init_scale',

'layer_params_per_param_grouping_exclude_preattnln___init_scale']

bash(f'mkdir -p {dest_dir}')
new_files = [f'{osp.join(dest_dir, init_id)}.sh' for init_id in init_ids]

for new_file in new_files:
  bash(f'cp {file_to_copy} {new_file}')