# parser.add_argument("--num_processes", type=int, required=True)
# parser.add_argument("--num_data_to_gen", type=int, required=True)
# parser.add_argument("--data_dir", type=str, required=True)
# parser.add_argument("--task", type=str, required=True)

# parser.add_argument("--task_params", type=str, required=True)
# parser.add_argument("--is_delete_existing_data_dir", type=str_to_bool, nargs='?', const=False, default=False)


# (num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time)
# python cfg_tasks/generate_cfg_task_data.py \
#   --num_processes 2 \
#   --num_data_to_gen 20 \
#   --data_dir cfg_tasks/generated_cfg_task_data/mar20/debug/induct \
#   --task induct \
#   --task_params 7,5,5,4,4,3 \
#   --is_delete_existing_data_dir True



# python cfg_tasks/generate_cfg_task_data.py \
#   --num_processes 2 \
#   --num_data_to_gen 20 \
#   --data_dir cfg_tasks/generated_cfg_task_data/mar20/debug/deduct \
#   --task deduct \
#   --task_params 7,5,5,4,4,3 \
#   --is_delete_existing_data_dir True


# num_rules, num_variables, num_terminals, max_rhs_len, depth, num_sents, timeout_time
  python cfg_tasks/generate_cfg_task_data.py \
  --num_processes 2 \
  --num_data_to_gen 20 \
  --data_dir cfg_tasks/generated_cfg_task_data/mar20/debug/abduct \
  --task abduct \
  --task_params 5,3,3,3,3,15,5 \
  --is_delete_existing_data_dir True