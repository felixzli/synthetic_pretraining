#!/bin/bash

MTN=cnndm_from_pretraining_with_nonsense_paper

CKPT=$1
EXP=$2
VM=$3
IDK=$4
init_with_custom_value=$5

LEARNING_RATE=0.001 
TRAIN_STEPS=90000
PERIOD=90000
EVAL_PERIOD=15000
INPUT_LEN=512
TARGET_LEN=256
METRIC=rouge1
EVALUATOR_NUM_EXAMPLES=6000

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
python extract_results/max_valid_acc_metric.py $exp_logdir $METRIC a
exit 1
fi

if [ "$IDK" = "mvam" ]
then
python extract_results/max_valid_acc_metric.py $exp_logdir $METRIC a
exit 1
fi


python ./t5x/train.py \
--gin_file=t5x/configs/final_exps/t5/finetune/100.gin \
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
--gin.init_with_custom_value="'$init_with_custom_value'" \
--gin.t5_small_t5x_checkpoint_path="'$CKPT'" \
--gin.EVALUATOR_NUM_EXAMPLES=$EVALUATOR_NUM_EXAMPLES \
--gin.shuffle_infer_eval=True \
--gin.JSON_WRITE_N_RESULTS=3 

echo /mnt/disks/persist/t5_training_models/${EXP}


# which_params_to_init_mean
# pre_self_attn_ln-pre_cross_attn_ln