#!/bin/bash

MTN=isarstep

CKPT=/mnt/disks/persist/t5_training_models/final_exps/nonlanguage_tasks/isarstep/from_scratch/best_lr_main_table/checkpoint_300000
# EXP=final_exps/nonlanguage_tasks/isarstep/runs/inference/infer/
VM=2
IDK=$1

is_load_everything_but_embed=True
if [ "$CKPT" = "from_scratch" ]
then
is_load_everything_but_embed=False
fi

LEARNING_RATE=0.0
TRAIN_STEPS=1
PERIOD=1
EVAL_PERIOD=1
INPUT_LEN=1024
TARGET_LEN=256
METRIC=em



EXP="${0%.*}"
exp_logdir=/mnt/disks/persist/t5_training_models/${EXP}

if [ "$IDK" = "echo_exp_logdir" ]
then
echo $exp_logdir
exit 1
fi

if [ "$IDK" = "remove_exp_logdir" ]
then
rm -rf $exp_logdir
exit 1
fi

if [ "$IDK" = "scp_results_no_preds" ]
then
bash bash_utils/copy_finetune_no_predictions.sh $EXP quantum2x2-$VM $MTN
fi


if [ "$IDK" = "scp" ]
then
bash bash_utils/copy_finetune.sh $EXP quantum2x2-$VM $MTN
fi


if [ "$IDK" = "max_valid_acc_metric" ]
then
python extract_results/max_valid_acc_metric.py $exp_logdir $METRIC
exit 1
fi
# rm -rf $exp_logdir

python ./t5x/train.py \
--gin_file=t5x/configs/final_exps/t5/finetune/small_isarstep_vocab.gin \
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
--gin.is_sanity_check_load_weights=True \
--gin.t5_small_t5x_checkpoint_path="'$CKPT'" \
--gin.EVALUATOR_NUM_EXAMPLES=5000 \
--gin.shuffle_infer_eval=False \
--gin.JSON_WRITE_N_RESULTS=5000

echo /mnt/disks/persist/t5_training_models/${EXP}

# /mnt/disks/persist/t5_training_models/final_exps/nonlanguage_tasks/isarstep/lime/best_lr_main_table/checkpoint_300000