#!/bin/bash


CKPT='/mnt/disks/persist/t5_training_models/final_exps/mar20_cfg/abduct_pretrain_t5small/checkpoint_524288'
EXP="${0%.*}"
VM=2


bash final_exps/mar20_cfg/finetune_first_runs/mtop/finetune_ARGS_checkpoint_exp_vm_idk.sh $CKPT $EXP $VM $1


# 
