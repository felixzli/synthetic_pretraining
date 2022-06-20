#!/bin/bash

MTN=code_trans_java_to_cs

CKPT=$1
EXP=$2
VM=$3
IDK=$4

is_load_everything_but_embed=True
if [ "$CKPT" = "from_scratch" ]
then
is_load_everything_but_embed=False
fi

LEARNING_RATE=0.0003 
TRAIN_STEPS=80000
PERIOD=30000
EVAL_PERIOD=5000
INPUT_LEN=1024
TARGET_LEN=1024
METRIC=bleu,em

exp_logdir=/mnt/disks/persist/t5_training_models/${EXP}

if [ "$IDK" = "echo_exp_logdir" ]
then
echo $exp_logdir
exit 1

fi

if [ "$IDK" = "remove_exp_logdir" ]
then
rm -rf $exp_logdir
fi

if [ "$IDK" = "scp_results_no_preds" ]
then
bash bash_utils/copy_finetune_no_inference.sh $EXP quantum2x2-$VM $MTN
fi

if [ "$IDK" = "max_valid_acc_metric" ]
then
python extract_results/max_valid_acc_metric.py $exp_logdir $METRIC
exit 1
fi

python ./t5x/train.py \
--gin_file=t5x/configs/final_exps/t5/finetune/small.gin \
--gin.TRAIN_STEPS=$TRAIN_STEPS \
--gin.INITIAL_CHECKPOINT_PATH=None \
--gin.RESTORE=None \
--gin.MIXTURE_OR_TASK_NAME="'${MTN}'" \
--gin.TASK_FEATURE_LENGTHS="{'inputs': $INPUT_LEN, 'targets': $TARGET_LEN}" \
--gin.BATCH_SIZE=32 \
--gin.PERIOD=$PERIOD \
--gin.EVAL_PERIOD=$EVAL_PERIOD \
--gin.LEARNING_RATE=$LEARNING_RATE \
--gin.PACK=False \
--gin.DROPOUT_RATE=0.1 \
--gin.FACTORS="'constant'" \
--gin.optimizer="@adafactor.Adafactor()" \
--gin.BEAM_SIZE=4 \
--gin.MAX_DECODE_LENGTH=1024 \
--gin.MODEL_DIR="'$exp_logdir'" \
--gin.is_load_everything_but_embed=$is_load_everything_but_embed \
--gin.t5_small_t5x_checkpoint_path="'$CKPT'" \
--gin.EVALUATOR_NUM_EXAMPLES=3000 \
--gin.override_infer_eval_dataset_cfg=None \
--gin.JSON_WRITE_N_RESULTS=3

echo /mnt/disks/persist/t5_training_models/${EXP}