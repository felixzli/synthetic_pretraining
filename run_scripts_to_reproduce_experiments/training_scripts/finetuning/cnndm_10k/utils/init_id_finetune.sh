#!/bin/bash

MTN=cnndm_10k
exp_logdir=$1
CKPT=$2
init_id=$3
IDK=$4

is_load_everything_but_embed=True
if [ "$CKPT" = "from_scratch" ]
then
is_load_everything_but_embed=False
fi

LEARNING_RATE=0.001 
TRAIN_STEPS=100000
PERIOD=10000
EVAL_PERIOD=10000
INPUT_LEN=512
TARGET_LEN=256
METRIC=rouge1
EVALUATOR_NUM_EXAMPLES=20000




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
--gin.MAX_DECODE_LENGTH=148 \
--gin.MODEL_DIR="'$exp_logdir'" \
--gin.init_id="'$init_id'" \
--gin.sanity_flag=False \
--gin.t5_small_t5x_checkpoint_path="'$CKPT'" \
--gin.EVALUATOR_NUM_EXAMPLES=$EVALUATOR_NUM_EXAMPLES \
--gin.JSON_WRITE_N_RESULTS=3



