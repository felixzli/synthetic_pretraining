
#!/bin/bash

_file_path_no_ext="${0%.*}"
file_path_no_ext_basename=${_file_path_no_ext##*/}
# file_path_no_ext_basename equals `27tasks`


_parent_dir="$(dirname "$0")"
parent_dir_basename=${_parent_dir##*/}

_parent_parent_dir="$(dirname "$_parent_dir")"
_parent_parent_dir_basename=${_parent_parent_dir##*/}
echo $_parent_parent_dir_basename
_parent_parent_parent_dir="$(dirname "$_parent_parent_dir")"
_parent_parent_parent_dir_basename=${_parent_parent_parent_dir##*/}
echo $_parent_parent_parent_dir_basename

# _parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_dir")"
# _parent_parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_parent_dir")"
# _parent_parent_parent_parent_parent_parent_dir="$(dirname "$_parent_parent_parent_parent_parent_dir")"


for N in lime copy set
do
  echo $N
	bash final_exps/language_tasks/$_parent_parent_dir_basename/runs/init_exps/$N/per_param_grouping___init_mean_std/run1.sh $1
done


