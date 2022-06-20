#!/bin/bash 
tasks=("copy" "reverse" "set" "first_char" "last_char" "length" "duplicate" "deduplicate" "longest_word")
num_tasks=${#tasks[@]}

save_folder=./data/1-4/
length_range=10,100
num_examples=1500000

# (trap 'kill 0' SIGINT;
for ((i=0;i<num_tasks;i+=1)); do
    python generate.py --task ${tasks[$i]} --save_folder $save_folder --length_range $length_range --num_examples $num_examples & 
done
wait
# )

wc -l ${save_folder}*/*

python data_scripts/1-4/split_1-4_data_train_val.py 1000000 4000 ./data/1-4/\*/deduped\*

bash data_scripts/1-4/save_data_to_gcs.sh
