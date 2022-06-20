#!/bin/bash

MTN=isarstep

CKPT=$1
EXP=$2
VM=$3
IDK=$4

is_load_everything_but_embed=True
if [ "$CKPT" = "from_scratch" ]
then
is_load_everything_but_embed=False
fi

LEARNING_RATE=0.003
TRAIN_STEPS=300000
PERIOD=50000
EVAL_PERIOD=50000
INPUT_LEN=1024
TARGET_LEN=256
METRIC=em


exp_logdir=/mnt/disks/persist/t5_training_models/${EXP}

if [ "$IDK" = "echo_exp_logdir" ]
then
echo $exp_logdir
fi

if [ "$IDK" = "remove_exp_logdir" ]
then
rm -rf $exp_logdir
fi

if [ "$IDK" = "scp_results_no_preds" ]
then
bash bash_utils/copy_finetune_no_predictions.sh $EXP quantum2x2-$VM $MTN
fi

if [ "$IDK" = "max_valid_acc_metric" ]
then
python extract_results/max_valid_acc_metric.py $exp_logdir $METRIC
# python extract_results/avg_metric.py $exp_logdir/inference_eval/$MTN-metrics.jsonl -1,0 $METRIC
echo 88888
exit 1
fi

# if [ "$1" = "9" ]
# then
# echo $EXP
# python extract_results/avg_metric.py $exp_logdir/inference_eval/$MTN-metrics.jsonl -1,0 accuracy
# exit 1
# fi


python ./t5x/train.py \
--gin_file=t5x/configs/final_exps/t5/finetune/small_isarstep_vocab.gin \
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
--gin.MAX_DECODE_LENGTH=256 \
--gin.MODEL_DIR="'$exp_logdir'" \
--gin.is_init_with_statistics=True \
--gin.t5_small_t5x_checkpoint_path="'$CKPT'" \
--gin.EVALUATOR_NUM_EXAMPLES=2500 \
--gin.shuffle_infer_eval=True \
--gin.JSON_WRITE_N_RESULTS=3

echo /mnt/disks/persist/t5_training_models/${EXP}