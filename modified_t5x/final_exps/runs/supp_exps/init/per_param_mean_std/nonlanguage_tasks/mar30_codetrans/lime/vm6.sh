#!/bin/bash

_file_path_no_ext="${0%.*}"
file_path_no_ext_basename=${_file_path_no_ext##*/}
# file_path_no_ext_basename equals `27tasks`


_parent_dir="$(dirname "$0")"
parent_dir_basename=${_parent_dir##*/}
echo std_exps_${parent_dir_basename}_${file_path_no_ext_basename} 

_parent_parent_dir="$(dirname "$_parent_dir")"
_parent_parent_dir_basename=${_parent_parent_dir##*/}
echo $_parent_parent_dir_basename
_parent_parent_parent_dir="$(dirname "$_parent_parent_dir")"
_parent_parent_parent_dir_basename=${_parent_parent_parent_dir##*/}
echo $_parent_parent_parent_dir_basename

# _parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_dir")"
# _parent_parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_parent_dir")"
# _parent_parent_parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_parent_parent_dir")"


for N in 1 2 3 4 
do
  echo $N
	bash final_exps/$_parent_parent_parent_dir_basename/$_parent_parent_dir_basename/runs/std_exps/diff_ckpt/$parent_dir_basename/run$N.sh $1
done

# bash final_exps/$_parent_parent_dir_basename/$parent_dir_basename/runs/std_exps/same_ckpt/from_scratch/run1.sh $1
# bash final_exps/$_parent_parent_dir_basename/$parent_dir_basename/runs/std_exps/same_ckpt/from_scratch/run2.sh $1
# bash final_exps/$_parent_parent_dir_basename/$parent_dir_basename/runs/std_exps/same_ckpt/from_scratch/run3.sh $1
# bash final_exps/$_parent_parent_dir_basename/$parent_dir_basename/runs/std_exps/same_ckpt/from_scratch/run4.sh $1
# bash final_exps/$_parent_parent_dir_basename/$parent_dir_basename/runs/std_exps/same_ckpt/from_scratch/run5.sh $1 