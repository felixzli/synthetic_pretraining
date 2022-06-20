# (num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time)
python cfg_tasks/generate_cfg_task_data.py \
  --num_processes 60 \
  --num_data_to_gen 1100000 \
  --data_dir cfg_tasks/generated_cfg_task_data/mar20/induct \
  --task induct \
  --task_params 7,5,5,4,4,5 \
  --is_delete_existing_data_dir $1


python cfg_tasks/generate_cfg_task_data.py \
  --num_processes 60 \
  --num_data_to_gen 1100000 \
  --data_dir cfg_tasks/generated_cfg_task_data/mar20/deduct \
  --task deduct \
  --task_params 7,5,5,4,4,5 \
  --is_delete_existing_data_dir $1


# num_rules, num_variables, num_terminals, max_rhs_len, depth, num_sents, timeout_time
  python cfg_tasks/generate_cfg_task_data.py \
  --num_processes 60 \
  --num_data_to_gen 1100000 \
  --data_dir cfg_tasks/generated_cfg_task_data/mar20/abduct \
  --task abduct \
  --task_params 4,3,3,3,3,25,5 \
  --is_delete_existing_data_dir $1