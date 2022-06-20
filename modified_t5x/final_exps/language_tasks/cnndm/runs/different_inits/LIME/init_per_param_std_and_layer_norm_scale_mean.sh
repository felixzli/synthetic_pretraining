#!/bin/bash

_file_path_no_ext="${0%.*}"
file_path_no_ext_basename=${_file_path_no_ext##*/}
echo $file_path_no_ext_basename 
# file_path_no_ext_basename equals `27tasks`


_parent_dir="$(dirname "$0")"
parent_dir_basename=${_parent_dir##*/}
echo $parent_dir_basename 

_parent_parent_dir="$(dirname "$_parent_dir")"

_parent_parent_parent_dir="$(dirname "$_parent_parent_dir")"

# parent_dir_basename equals `apr1_first_try``

python final_exps/final_exps_utils/finetune/finetune.py \
--exp_suffix $file_path_no_ext_basename  \
--pretrain_id lime \
--vm 4 \
--idk $1 --finetune_bash_file final_exps/language_tasks/cnndm/runs/different_inits/LIME/_init_per_param_std_and_layer_norm_scale_mean.sh

