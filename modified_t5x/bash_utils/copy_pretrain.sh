#!/bin/bash
EXP_NAME=$1
synthetic_task_name=$3

BASE_SAVE_DIR=/Users/felix/Documents/research/results_and_analysis/universal_lime/$EXP_NAME
BASE_RESULTS_DIR=/mnt/disks/persist/t5_training_models/$EXP_NAME


VALID_SAVE_DIR=$BASE_SAVE_DIR/valid
TRAIN_SAVE_DIR=$BASE_SAVE_DIR/train


mkdir -p $VALID_SAVE_DIR
gcloud compute firewall-rules create --network="default" allow-ssh --allow=tcp:22
gcloud alpha compute tpus tpu-vm scp --recurse \
$2:$BASE_RESULTS_DIR/training_eval/*/ \
$VALID_SAVE_DIR --zone us-central1-a


mkdir -p $TRAIN_SAVE_DIR
gcloud compute firewall-rules create --network="default" allow-ssh --allow=tcp:22
gcloud alpha compute tpus tpu-vm scp \
$2:$BASE_RESULTS_DIR/train/events* \
$TRAIN_SAVE_DIR --zone us-central1-a


# /mnt/disks/persist/t5_training_models/reproduce_pwn/12-25/pretrain_lr1e-4



# gcloud compute firewall-rules create --network="default" allow-ssh --allow=tcp:22
# gcloud alpha compute tpus tpu-vm scp --recurse \
# quantum2x2-6:/mnt/disks/persist/felix-text-to-text-transfer-transformer/tmp/*pack* \
# /Users/felix/Documents/research/results_and_analysis/universal_lime/synthetic_tasks/investigate_mixture_pack --zone us-central1-a
