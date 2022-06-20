import argparse
import os
import os.path as osp


def split_data_train_val(data_path, train_count, valid_count):
  data_dir = osp.dirname(data_path)
  data_file_name = osp.basename(data_path)
  train_path = osp.join(data_dir, f'train{train_count}_' + data_file_name)
  valid_path = osp.join(data_dir, f'valid{valid_count}_' + data_file_name)
  os.system(f'head -n {train_count} {data_path} > {train_path}')
  os.system(f'tail -n {valid_count} {data_path} > {valid_path}')
  return train_path, valid_path


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
  parser.add_argument('--data_path', type=str, default=None, required=True)
  parser.add_argument('--train_count', type=int, default=None, required=True)
  parser.add_argument('--valid_count', type=int, default=None, required=True)
  # parser.add_argument('--some_boolean', type=str_to_bool, nargs='?', const=True, default=False)
  args = parser.parse_args()
  
  train_path, valid_path = split_data_train_val(args.data_path, args.train_count, args.valid_count)
  os.system(f'head -n 2 {train_path} {valid_path}')
  os.system(f'wc -l {train_path} {valid_path}')

