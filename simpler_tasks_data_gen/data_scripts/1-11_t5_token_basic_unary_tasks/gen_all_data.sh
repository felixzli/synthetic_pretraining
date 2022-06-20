#!/bin/bash 
tasks=("copy" "reverse" "set" "first_char" "last_char" "length" "duplicate" "deduplicate" "longest_word")
num_tasks=${#tasks[@]}

parent_folder=$(basename $(dirname "$0"))/
echo parent_folder
save_folder=./data/$parent_folder
length_range=10,100
num_examples=120000

# (trap 'kill 0' SIGINT;
for ((i=0;i<num_tasks;i+=1)); do
    python generate.py \
    --task ${tasks[$i]} \
    --save_folder $save_folder \
    --is_tasks_with_tokens True \
    --length_range $length_range \
    --num_examples $num_examples & 
done
wait
# )

wc -l ${save_folder}/*/*

python data_scripts/split_all_data_train_val.py 100000 10000 ./data/$parent_folder/\*/deduped\*

bash data_scripts/save_data_to_gcs.sh $parent_folder