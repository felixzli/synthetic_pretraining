#!/bin/bash


# exp_dir === path to directory where you want training checkpoints and results saved
exp_dir=/mnt/disks/persist/test_final_code/

# task_id === name of task to train on

# choices for task_id: 
# lime, nonsense_summary, nesting_language, 
# set, identity
# delete, sort, union, replace, duplicate, intersect, reverse, 
# deduplicate, search, longest_word, length, count, first_token, last_token, set1_minus_set2, set2_minus_set1

task_id=lime


























MTN=$task_id
if [ "$1" = "mas" ]
then
python modified_t5x/extract_results/get_max_acc_step_given_exp_dir.py $exp_dir
exit 1
fi

if [ "$1" = "99" ]
then
python modified_t5x/extract_results/get_99_acc_step_given_exp_dir.py $exp_dir
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


python t5x/train.py \
--gin_file=t5x/configs/synthetic_tasks/1-22/pretrain/t5_small_AND_pretrain.gin \
--gin.TRAIN_STEPS=80000  \
--gin.INITIAL_CHECKPOINT_PATH="'askjdnsaljdnkajsnd'" \
--gin.RESTORE=None \
--gin.MIXTURE_OR_TASK_NAME="'$MTN'" \
--gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets':512}" \
--gin.BATCH_SIZE=128 \
--gin.PERIOD=5000 \
--gin.EVAL_PERIOD=5000 \
--gin.LEARNING_RATE=1231237198 \
--gin.PACK=False \
--gin.DROPOUT_RATE=0.1 \
--gin.FACTORS="'rsqrt_decay'" \
--gin.optimizer="@adafactor.Adafactor()" \
--gin.MODEL_DIR="'$exp_dir'" \
--gin.is_init_with_statistics=False \
--gin.t5_small_t5x_checkpoint_path="'asdfsadfszdafsafasdfasdfa'" 

echo /mnt/disks/persist/t5_training_models/${EXP}

