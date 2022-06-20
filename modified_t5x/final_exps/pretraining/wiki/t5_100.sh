#!/bin/bash

LEARNING_RATE=182379128.0
EXP="${0%.*}"
MTN=wiki40b

if [ "$1" = "8" ]
then
bash bash_utils/copy_pretrain.sh $EXP quantum2x2-2 $MTN
fi

exp_dir=/mnt/disks/persist/t5_training_models/${EXP}
if [ "$1" = "mas" ]
then
python extract_results/get_max_acc_step_given_exp_dir.py $exp_dir
exit 1
fi

if [ "$1" = "99" ]
then
python extract_results/get_99_acc_step_given_exp_dir.py $exp_dir
exit 1
fi

if [ "$1" = "remove" ]
then
rm -rf $exp_dir
echo removed exp_dir
exit 1
fi

if [ "$1" = "exp_dir" ]
then
echo ECHO EXP DIR
echo $exp_dir
exit 1
fi


python ./t5x/train.py \
--gin_file=t5x/configs/final_exps/t5/pretrain/t5_100.gin \
--gin.TRAIN_STEPS=150000  \
--gin.INITIAL_CHECKPOINT_PATH="'askjdnsaljdnkajsnd'" \
--gin.RESTORE=None \
--gin.MIXTURE_OR_TASK_NAME="'$MTN'" \
--gin.TASK_FEATURE_LENGTHS="{'inputs': 256, 'targets':256}" \
--gin.BATCH_SIZE=32 \
--gin.PERIOD=10000 \
--gin.EVAL_PERIOD=10000 \
--gin.LEARNING_RATE=$LEARNING_RATE \
--gin.PACK=False \
--gin.DROPOUT_RATE=0.1 \
--gin.FACTORS="'rsqrt_decay'" \
--gin.optimizer="@adafactor.Adafactor()" \
--gin.MODEL_DIR="'/mnt/disks/persist/t5_training_models/${EXP}'" \
--gin.is_init_with_statistics=False \
--gin.t5_small_t5x_checkpoint_path="'asdfsadfszdafsafasdfasdfa'" 


echo /mnt/disks/persist/t5_training_models/${EXP}

