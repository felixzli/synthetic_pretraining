import sys
import os.path as osp
from os import system as bash



file_to_copy = 'final_exps/language_tasks/cnndm/runs/1final_init_exps/LIME/test.sh'
dest_dir = 'final_exps/language_tasks/cnndm/runs/1final_init_exps/LIME_include_nonlayer_params/'
init_ids = [

# per param grouping (include nonlayer params)
'per_param_grouping___init_scale', 

  
# across layer grouping (include nonlayer params)
'across_layer_grouping___init_scale', 
'across_layer_big_grouping___init_scale', 

# nonlayer params
'nonlayer_params_per_param_grouping___init_scale', 
'nonlayer_ln_per_param_grouping___init_scale', 
'relpos_per_param_grouping___init_scale', 
'token_embed_per_param_grouping___init_scale',

#####
'per_param_grouping_exclude_relpos___init_scale',
'per_param_grouping_exclude_relpos_and_preattnln___init_scale',
'relpos_together_grouping___init_scale',
'relpos_together_and_preattnln_across_layer_grouping___init_scale'
]

bash(f'mkdir -p {dest_dir}')
new_files = [f'{osp.join(dest_dir, init_id)}.sh' for init_id in init_ids]

for new_file in new_files:
  bash(f'cp {file_to_copy} {new_file}')