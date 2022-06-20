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

# exp_dir=/mnt/disks/persist/t5_training_models/final_exps/nonlanguage_tasks/isarstep/from_scratch/from_scratch_learning_rate_sweep/lr1e-3
# rm -rf $exp_dir
# cp -r /mnt/disks/persist/t5_training_models/final_exps/nonlanguage_tasks/isarstep/from_scratch/from_scratch_learning_rate_sweep $exp_dir
# ls $exp_dir
python final_exps/final_exps_utils/finetune/finetune.py \
--exp_suffix ${parent_dir_basename}/${file_path_no_ext_basename} \
--pretrain_id from_scratch \
--vm 5 \
--idk $1 --finetune_bash_file ${_parent_parent_parent_dir}/_finetune_${file_path_no_ext_basename}.sh


