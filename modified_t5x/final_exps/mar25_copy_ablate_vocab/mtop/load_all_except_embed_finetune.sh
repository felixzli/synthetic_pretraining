
#!/bin/bash

MTN=mtop

CKPT=$1
EXP=$2
VM=$3
IDK=$4

LEARNING_RATE=0.001 
TRAIN_STEPS=60000
PERIOD=30000
EVAL_PERIOD=5000
INPUT_LEN=1024
TARGET_LEN=128
METRIC=em




if [ "$IDK" = "9" ]
then
echo fooooo
echo $EXP
python extract_results/avg_metric.py /mnt/disks/persist/t5_training_models/${EXP}/inference_eval/$MTN-metrics.jsonl 3,0 $METRIC
exit 1
fi


if [ "$IDK" = "6" ]
then
echo $EXP
fi
# rm -rf /mnt/disks/persist/t5_training_models/$EXP

if [ "$IDK" = "8" ]
then
bash bash_utils/copy_finetune_no_predictions.sh $EXP quantum2x2-$VM $MTN
fi
# bash synthetic_tasks/1-14/get_freeze_load_copy1e-4_pretrain_checkpoint.sh

exp_logdir=/mnt/disks/persist/t5_training_models/${EXP}
if [ "$IDK" = "888" ]
then
python extract_results/max_valid_acc_metric.py $exp_logdir em
exit 1
fi


python ./t5x/train.py \
--gin_file=t5x/configs/synthetic_tasks/1-22/finetune/t5_small_AND_finetune.gin \
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
--gin.MAX_DECODE_LENGTH=128 \
--gin.MODEL_DIR="'/mnt/disks/persist/t5_training_models/${EXP}'" \
--gin.is_sanity_check_load_weights=False \
--gin.is_load_everything_but_embed=True \
--gin.t5_small_t5x_checkpoint_path="'$CKPT'" \
--gin.EVALUATOR_NUM_EXAMPLES=1500 \
--gin.JSON_WRITE_N_RESULTS=3

echo /mnt/disks/persist/t5_training_models/${EXP}