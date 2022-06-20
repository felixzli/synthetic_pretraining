#!/bin/bash 
# tasks=("longest_word" "copy" "reverse" "set" "first_char" "last_char" "length" "duplicate" "deduplicate")
# tasks=("first_char" "last_char")
tasks=("copy")


num_tasks=${#tasks[@]}

parent_folder=$(basename $(dirname "$0"))
echo parent_folder
save_folder=./data/$parent_folder
length_range=252,252
num_examples=1020000


if [ "$1" = "g" ]
then
bash data_scripts/get_data_from_gcs.sh $parent_folder
exit 1
fi



(trap 'kill 0' SIGINT;
for ((i=0;i<num_tasks;i+=1)); do
    python generate.py \
    --task ${tasks[$i]} \
    --save_folder $save_folder \
    --is_tasks_with_tokens True \
    --num_tokens 10 \
    --length_range $length_range \
    --num_examples $num_examples & 
done
wait
)

wc -l ${save_folder}/*/*

python data_scripts/split_all_data_train_val.py 1000000 10000 ./data/$parent_folder/\*/deduped\*

# bash data_scripts/$parent_folder/cp_data_to_fttttt.sh

bash data_scripts/save_data_to_gcs.sh $parent_folder

