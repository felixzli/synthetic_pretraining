#!/bin/bash

_file_path_no_ext="${0%.*}"
file_path_no_ext_basename=${_file_path_no_ext##*/}
echo ------------------------------------------------
echo $_file_path_no_ext
# file_path_no_ext_basename equals `27tasks`


_parent_dir="$(dirname "$0")"
parent_dir_basename=${_parent_dir##*/}
# echo $parent_dir_basename 

_parent_parent_dir="$(dirname "$_parent_dir")"
parent_parent_dir_basename=${_parent_parent_dir##*/}

_parent_parent_parent_dir="$(dirname "$_parent_parent_dir")"
_parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_dir")"
_parent_parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_parent_dir")"
_parent_parent_parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_parent_parent_dir")"


# parent_dir_basename equals `apr1_first_try``
echo  $parent_dir_basename
python final_exps/final_exps_utils/finetune/finetune.py \
--exp_suffix $_file_path_no_ext  \
--pretrain_id $parent_parent_dir_basename \
--vm 2 \
--idk $1 --finetune_bash_file $_parent_parent_parent_dir/_init_finetune.sh --init_id $parent_dir_basename