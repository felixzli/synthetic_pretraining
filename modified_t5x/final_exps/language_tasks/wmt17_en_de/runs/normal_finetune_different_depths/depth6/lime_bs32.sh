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
--exp_suffix $_file_path_no_ext \
--pretrain_id lime \
--vm 2 \
--idk $1 --finetune_bash_file final_exps/language_tasks/wmt17_en_de/runs/normal_finetune_different_depths/depth6/_bs32.sh

