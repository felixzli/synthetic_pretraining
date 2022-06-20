#!/bin/bash

exp_dir=/mnt/disks/persist/debug/
ckpt=/mnt/disks/persist/t5_training_models/CHECKPOINTS_FROM_GCS/feb13_t5_small_ckpts/copy/checkpoint/
init_id=per_param_grouping___init_mean_std


_parent_dir="$(dirname "$0")"
parent_dir_basename=${_parent_dir##*/}
_parent_parent_dir="$(dirname "$_parent_dir")"
parent_parent_dir_basename=${_parent_parent_dir##*/}
_parent_parent_parent_dir="$(dirname "$_parent_parent_dir")"
parent_parent_parent_dir_basename=${_parent_parent_parent_dir##*/}


bash ../run_scripts_to_reproduce_experiments/training_scripts/finetuning/$parent_dir_basename/utils/init_id_finetune.sh $exp_dir $ckpt $init_id