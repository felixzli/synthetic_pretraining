import argparse
import os
import os.path as osp


def convert_to_flax_data(data_path, new_flax_data_src_path, new_flax_data_tgt_path):
  assert 'src' in new_flax_data_src_path
  assert 'tgt' in new_flax_data_tgt_path

  with open(data_path, 'r') as data, \
    open(new_flax_data_src_path, 'w') as src, \
    open(new_flax_data_tgt_path, 'w') as tgt:
    for line in data:
      input, output = line.split('\t')
      s = input.split(' ')
      input_task, input_rest = s[0], ' '.join(s[1:])
      input_rest = input_rest.replace(' ', '')
      input = input_task + ' ' + input_rest
      output = output.replace(' ', '')
      src.write(input+'\n')
      tgt.write(output)


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
  parser.add_argument('--new_flax_data_src_path', type=str, default=None, required=True)
  parser.add_argument('--new_flax_data_tgt_path', type=str, default=None, required=True)

  args = parser.parse_args()
  
  data_path = args.data_path
  new_flax_data_src_path = args.new_flax_data_src_path
  new_flax_data_tgt_path = args.new_flax_data_tgt_path
  os.makedirs(osp.dirname(new_flax_data_src_path), exist_ok=True)
  convert_to_flax_data(data_path, new_flax_data_src_path, new_flax_data_tgt_path)
  os.system(f'head -n 2 {data_path} {new_flax_data_src_path} {new_flax_data_tgt_path}')
  os.system(f'wc -l {data_path} {new_flax_data_src_path} {new_flax_data_tgt_path}')


