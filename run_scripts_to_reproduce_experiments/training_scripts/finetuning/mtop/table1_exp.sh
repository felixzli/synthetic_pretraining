#!/bin/bash

exp_dir=/mnt/disks/persist/debug/
ckpt=from_scratch


_parent_dir="$(dirname "$0")"
parent_dir_basename=${_parent_dir##*/}
_parent_parent_dir="$(dirname "$_parent_dir")"
parent_parent_dir_basename=${_parent_parent_dir##*/}
_parent_parent_parent_dir="$(dirname "$_parent_parent_dir")"
parent_parent_parent_dir_basename=${_parent_parent_parent_dir##*/}

bash ../run_scripts_to_reproduce_experiments/training_scripts/finetuning/$parent_dir_basename/utils/finetune.sh $exp_dir $ckpt