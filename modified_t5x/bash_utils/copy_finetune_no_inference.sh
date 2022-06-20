#!/bin/bash

EXP_NAME=$1
# EXP_NAME=/12-25/lowercase_finetune_bs16_lr1e-2_pretrain12-25

BASE_SAVE_DIR=/Users/felix/Documents/research/results_and_analysis/universal_lime/$EXP_NAME/
BASE_RESULTS_DIR=/mnt/disks/persist/t5_training_models/$EXP_NAME/

if [ "$3" != "" ]; then
  MTN=$3
else
  MTN=cnndm_from_pretraining_with_nonsense_paper
fi


VALID_SAVE_DIR=$BASE_SAVE_DIR/valid
TRAIN_SAVE_DIR=$BASE_SAVE_DIR/train


mkdir -p $VALID_SAVE_DIR
gcloud compute firewall-rules create --network="default" allow-ssh --allow=tcp:22
gcloud alpha compute tpus tpu-vm scp --recurse \
$2:$BASE_RESULTS_DIR/training_eval/*/ \
$VALID_SAVE_DIR --zone us-central1-a

# gcloud alpha compute tpus tpu-vm scp \
# $2:$BASE_RESULTS_DIR/inference_eval/*metrics.jsonl \
# $VALID_SAVE_DIR --zone us-central1-a

mkdir -p $TRAIN_SAVE_DIR
gcloud compute firewall-rules create --network="default" allow-ssh --allow=tcp:22
gcloud alpha compute tpus tpu-vm scp \
$2:$BASE_RESULTS_DIR/train/events* \
$TRAIN_SAVE_DIR --zone us-central1-a


# /mnt/disks/persist/t5_training_models/reproduce_pwn/12-25/pretrain_lr1e-4

# /mnt/disks/persist/t5_training_models/reproduce_pwn/12-25/lowercase_finetune_bs16_lr1e-2/inference_eval/cnndm_from_pretraining_with_nonsense_paper-metrics.jsonl
