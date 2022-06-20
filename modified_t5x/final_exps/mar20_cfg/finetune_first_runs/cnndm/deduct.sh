#!/bin/bash


CKPT='/mnt/disks/persist/t5_training_models/final_exps/mar20_cfg/deduct_pretrain_t5small/checkpoint_25000'
EXP="${0%.*}"
VM=3


bash final_exps/mar20_cfg/finetune_first_runs/cnndm/finetune_ARGS_checkpoint_exp_vm_idk.sh $CKPT $EXP $VM $1