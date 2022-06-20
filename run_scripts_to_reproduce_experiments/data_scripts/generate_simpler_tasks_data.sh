#!/bin/bash 
# copy is the task referred to as identity in the paper
# first char and last char are the tasks referred to as first token and last token in the paper
tasks=('set' 'copy' 'delete' 'sort' 'union' 'set_1_minus_2' 'set_2_minus_1' 'replace' 'duplicate' 'intersect' 'reverse' \
      'deduplicate' 'last_char' 'first_char' 'search' 'longest_word' 'length' 'count')
num_tasks=${#tasks[@]}
save_folder=data/generated_data/
length_range=10,220
num_examples=1050000

(trap 'kill 0' SIGINT;
for ((i=0;i<num_tasks;i+=1)); do
    echo $i
    python simpler_tasks_generation/generate.py \
    --task ${tasks[$i]} \
    --save_folder $save_folder \
    --is_tasks_with_tokens True \
    --length_range $length_range \
    --num_examples $num_examples 
done
wait
)

wc -l ${save_folder}/*/*

python simpler_tasks_data_gen/data_scripts/split_all_data_train_val.py 1000000 10000 $save_folder/\*/deduped\*