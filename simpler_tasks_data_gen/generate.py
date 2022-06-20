import argparse
import os.path as osp
import random


def deduplicate_generated_data(data_path):
  data_dir = osp.dirname(data_path)
  dedup_data_path = osp.join(data_dir, 'deduped_' + osp.basename(data_path))
  data_inputs_set = set()
  with open(data_path, 'r') as data_file, open(dedup_data_path, 'w') as dedup_data_file:
    for line in data_file:
      input = line.split('\t')[0]
      if input in data_inputs_set:
        continue
      data_inputs_set.add(input)
      dedup_data_file.write(line)


def str_to_bool(value):
  if isinstance(value, bool):
      return value
  if value.lower() in {'false', 'f', '0', 'no', 'n'}:
      return False
  elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
      return True
  raise ValueError(f'{value} is not a valid boolean value')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--task', type=str, default=None, required=True)
  parser.add_argument('--save_folder', type=str, default=None, required=True)
  parser.add_argument('--task_config_str', type=str, default=None, required=False)
  parser.add_argument('--length_range', type=str, default=None, required=True)
  parser.add_argument('--num_examples', type=int, default=None, required=True)
  parser.add_argument('--num_tokens', type=int, default=None, required=False)


  parser.add_argument('--is_tasks_with_tokens', type=str_to_bool, nargs=1, required=True)

  # parser.add_argument('--some_boolean', type=str_to_bool, nargs='?', const=True, default=False)
  args = parser.parse_args()
  num_tokens = args.num_tokens
  if args.is_tasks_with_tokens:
    from tasks_with_tokens import TASK_REGISTRY, generate_batch_examples, T5_TOKEN_IDS_FOR_TASKS

  else:
    from prototypical_tasks import TASK_REGISTRY, generate_batch_examples, UPPER_AND_LOWER_LETTERS

  task = args.task
  length_range = [int(n) for n in args.length_range.split(',')]
  assert len(length_range) == 2
  task_config_str = args.task_config_str
  if task_config_str is None:
    task_config_str = f"{task}{length_range[0]}_{length_range[1]}"
  if num_tokens is not None:
    task_config_str = f"{task}length{length_range[0]}_{length_range[1]}_tokens{num_tokens}"

  if 'lime' in task:
    lime_sampling_args = {'length_range':length_range, 'token_ids':list(T5_TOKEN_IDS_FOR_TASKS),'is_original_vocab_division': True,
                      'ida_num_chars_range':None, 'ida_pattern_length_range':None, 
                      'num_upper_case_letters':None, 'num_lower_case_letters':None, 'num_math_symbols':None}
    data_path = generate_batch_examples(task=TASK_REGISTRY[task], file_dir=args.save_folder, num_examples=args.num_examples, 
                        is_natural_language=False, task_config_str=task_config_str, **lime_sampling_args)
  elif args.is_tasks_with_tokens:
    if num_tokens is None:
      token_ids = T5_TOKEN_IDS_FOR_TASKS
    else:
      token_ids = random.sample(T5_TOKEN_IDS_FOR_TASKS, k=num_tokens)
    data_path = generate_batch_examples(task=TASK_REGISTRY[task], file_dir=args.save_folder, num_examples=args.num_examples, 
                          is_natural_language=False, task_config_str=task_config_str, length_range=length_range, token_ids=token_ids)
  else:
    data_path = generate_batch_examples(task=TASK_REGISTRY[task], file_dir=args.save_folder, num_examples=args.num_examples, 
                          is_natural_language=False, task_config_str=task_config_str, length_range=length_range, char_set=UPPER_AND_LOWER_LETTERS)
  deduplicate_generated_data(data_path)
