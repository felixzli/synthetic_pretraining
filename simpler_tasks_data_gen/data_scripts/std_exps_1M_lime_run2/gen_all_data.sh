#!/bin/bash 
tasks=("lime_abduct" "lime_deduct" "lime_induct")
num_tasks=${#tasks[@]}

parent_folder=$(basename $(dirname "$0"))
echo parent_folder
save_folder=./data/$parent_folder
length_range=0,220
num_examples=1050000

(trap 'kill 0' SIGINT;
for ((i=0;i<num_tasks;i+=1)); do
    python generate.py \
    --task ${tasks[$i]} \
    --save_folder $save_folder \
    --is_tasks_with_tokens True \
    --length_range $length_range \
    --num_examples $num_examples & 
done
wait
)

wc -l ${save_folder}/*/*

python data_scripts/split_all_data_train_val.py 1000000 10000 ./data/$parent_folder/\*/deduped\*

bash data_scripts/save_data_to_gcs.sh $parent_folder

# bash data_scripts/$parent_folder/cp_data_to_fttttt.sh


# bash data_scripts/std_exps_1M_lime_run1/gen_all_data.sh && data_scripts/std_exps_1M_lime_run2/gen_all_data.sh && bash data_scripts/std_exps_1M_lime_run3/gen_all_data.sh && bash data_scripts/std_exps_1M_lime_run4/gen_all_data.sh