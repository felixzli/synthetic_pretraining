#!/bin/bash

LEARNING_RATE=182379128.0
EXP="${0%.*}"
MTN=mar21_cfg_mix

if [ "$1" = "8" ]
then
bash bash_utils/copy_pretrain.sh $EXP quantum2x2-4 $MTN
fi
# rm -rf /mnt/disks/persist/t5_training_models/$EXP

exp_dir=/mnt/disks/persist/t5_training_models/${EXP}
if [ "$1" = "mas" ]
then
python extract_results/get_max_acc_step_given_exp_dir.py $exp_dir
exit 1
fi


python ./t5x/train.py \
--gin_file=t5x/configs/synthetic_tasks/1-22/pretrain/t5_small_AND_pretrain.gin \
--gin.TRAIN_STEPS=524288  \
--gin.INITIAL_CHECKPOINT_PATH="'askjdnsaljdnkajsnd'" \
--gin.RESTORE=None \
--gin.MIXTURE_OR_TASK_NAME="'$MTN'" \
--gin.TASK_FEATURE_LENGTHS="{'inputs': 1024, 'targets': 1024}" \
--gin.BATCH_SIZE=128 \
--gin.PERIOD=5000 \
--gin.EVAL_PERIOD=5000 \
--gin.LEARNING_RATE=$LEARNING_RATE \
--gin.PACK=True \
--gin.DROPOUT_RATE=0.1 \
--gin.FACTORS="'rsqrt_decay'" \
--gin.optimizer="@adafactor.Adafactor()" \
--gin.MODEL_DIR="'/mnt/disks/persist/t5_training_models/${EXP}'" \
--gin.is_init_with_statistics=False \
--gin.t5_small_t5x_checkpoint_path="'asdfsadfszdafsafasdfasdfa'" 


echo /mnt/disks/persist/t5_training_models/${EXP}

