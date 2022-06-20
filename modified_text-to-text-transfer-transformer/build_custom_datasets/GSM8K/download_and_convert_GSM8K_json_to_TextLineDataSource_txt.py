import os
import sys
import json
import os.path as osp
sys.path.append(os.getcwd())
from build_custom_datasets.utils import wc_files, head_files


dir_path = os.path.dirname(os.path.relpath(__file__))
json_data_folder = os.path.join(dir_path, 'gsm8k_json_data')
tlds_data_folder = 'downstream_tasks_data/gsm8k'


def convert_to_tlds(json_f, tlds_f):
  for l in json_f:
    data = json.loads(l)
    src = data['question'].replace('\n', ' ').replace('<<', '[[').replace('>>', ']]')
    tgt = data['answer'].replace('\n', ' ').replace('<<', '[[').replace('>>', ']]')
    tlds_f.write(f'{src}\t{tgt}\n')


if __name__ == '__main__':

  os.system(f'mkdir -p {tlds_data_folder}')
  os.system(f'mkdir -p {json_data_folder}')

  test_curl_path = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
  train_curl_path = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl'
  print(json_data_folder)
  os.system(f'(cd {json_data_folder} && \
            curl -O {test_curl_path} && curl -O {train_curl_path})')
  
  json_train_file = open(osp.join(json_data_folder, 'train.jsonl'), mode='r')
  json_test_file = open(osp.join(json_data_folder, 'test.jsonl'), mode='r')

  tlds_train_file = open(osp.join(tlds_data_folder, 'train.txt'), mode='w')
  tlds_valid_file = open(osp.join(tlds_data_folder, 'valid.txt'), mode='w')

  convert_to_tlds(json_train_file, tlds_train_file)
  convert_to_tlds(json_test_file, tlds_valid_file)

  tlds_train_file.close()
  tlds_valid_file.close()

  wc_files([json_train_file, json_test_file, tlds_train_file, tlds_valid_file])
  head_files([json_train_file, json_test_file, tlds_train_file, tlds_valid_file], 1)


  


