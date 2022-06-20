#!/bin/bash 
tasks=("copy" "reverse" "set" "first_char" "last_char" "length" "duplicate" "deduplicate" "longest_word")
num_tasks=${#tasks[@]}

save_folder=./data/11-29/
length_range=10,100
num_examples=2000000

# (trap 'kill 0' SIGINT;
for ((i=0;i<num_tasks;i+=1)); do
    python generate.py --task ${tasks[$i]} --save_folder $save_folder --length_range $length_range --num_examples $num_examples & 
done
wait
# )

wc -l ${save_folder}*/*

python data_scripts/11-29_data/split_all_data_train_val.py
python data_scripts/11-29_data/convert_train_val_data_to_flax_data.py
