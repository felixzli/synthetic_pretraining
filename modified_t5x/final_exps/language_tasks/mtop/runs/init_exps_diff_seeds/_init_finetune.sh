#!/bin/bash

MTN=mtop
init_id=$5

CKPT=$1
EXP=$2
VM=$3
IDK=$4

is_load_everything_but_embed=True
if [ "$CKPT" = "from_scratch" ]
then
is_load_everything_but_embed=False
fi

LEARNING_RATE=0.001 
TRAIN_STEPS=100000
PERIOD=30000
EVAL_PERIOD=10000
INPUT_LEN=1024
TARGET_LEN=128
METRIC=em,f1
EVALUATOR_NUM_EXAMPLES=5000

exp_logdir=/mnt/disks/persist/t5_training_models/${EXP}

if [ "$IDK" = "echo_exp_logdir" ]
then
echo $exp_logdir
fi

if [ "$IDK" = "remove_exp_logdir" ]
then
echo remove_exp_logdir
rm -rf $exp_logdir
exit 1
fi

if [ "$IDK" = "scp_results_no_preds" ]
then
echo $IDK
bash bash_utils/copy_finetune_no_predictions.sh $EXP quantum2x2-$VM $MTN
exit 1
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
--gin.BATCH_SIZE=128 \
--gin.PERIOD=$PERIOD \
--gin.EVAL_PERIOD=$EVAL_PERIOD \
--gin.LEARNING_RATE=$LEARNING_RATE \
--gin.PACK=False \
--gin.DROPOUT_RATE=0.1 \
--gin.FACTORS="'constant'" \
--gin.optimizer="@adafactor.Adafactor()" \
--gin.BEAM_SIZE=4 \
--gin.MAX_DECODE_LENGTH=$TARGET_LEN \
--gin.MODEL_DIR="'$exp_logdir'" \
--gin.init_id="'$init_id'" \
--gin.sanity_flag=False \
--gin.t5_small_t5x_checkpoint_path="'$CKPT'" \
--gin.EVALUATOR_NUM_EXAMPLES=$EVALUATOR_NUM_EXAMPLES \
--gin.JSON_WRITE_N_RESULTS=3

echo /mnt/disks/persist/t5_training_models/${EXP}


