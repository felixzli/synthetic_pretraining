import os
import os.path as osp
import glob
from os import system as bash


nonsense_summary_base = '/mnt/disks/persist/felix-text-to-text-transfer-transformer/pretraining_with_nonsense_paper_data/nonsense_summary_tasks'
cnndm_base = '/mnt/disks/persist/felix-text-to-text-transfer-transformer/pretraining_with_nonsense_paper_data/cnndm'

sem_parse_base = '/mnt/disks/persist/felix-text-to-text-transfer-transformer/downstream_tasks_data/'
retro_base = '/mnt/disks/persist/felix-text-to-text-transfer-transformer/t5/data/mar31_and_after_tasks/chemistry_data/single_step_retrosynthesis/converted_data_ready_for_seqio'
code_trans_base = '/mnt/disks/persist/felix-codexglue/convert_data_to_src_tab_tgt_txt_file/converted_data/code_trans_java_to_cs'
DOWNSTREAM_TASK_ID_TO_PATHS = {  
  'cnndm_10k': [f'{cnndm_base}/lowercase_train.txt', f'{cnndm_base}/lowercase_val.txt', '/mnt/disks/persist/felix-text-to-text-transfer-transformer/pretraining_with_nonsense_paper_data/dataset_root/finetuning_datasets/cnndm/lowercase_test.txt'],
  'mtop': [osp.join(sem_parse_base, 'mtop', x) for x in ['train.txt', 'valid.txt', 'test.txt']],
  'webqsp': [osp.join(sem_parse_base, 'webqsp', x) for x in ['train.txt', 'valid.txt', 'test.txt']],
  'retrosynthesis': [osp.join(retro_base, x) for x in ['train.txt', 'valid.txt', 'test.txt']],
  'code_translation': [osp.join(code_trans_base, x) for x in ['train.txt', 'valid.txt', 'test.txt']],
}


vm6_lime_paths = {
        "abduct_train": f"/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/std_exps_1M_lime_run1/lime_abduct/train1000000_deduped_lime_abduct0_220_1M.txt",
        "abduct_valid": f"/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/std_exps_1M_lime_run1/lime_abduct/valid10000_deduped_lime_abduct0_220_1M.txt",
        "deduct_train": f"/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/std_exps_1M_lime_run1/lime_deduct/train1000000_deduped_lime_deduct0_220_1M.txt",
        "deduct_valid": f"/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/std_exps_1M_lime_run1/lime_deduct/valid10000_deduped_lime_deduct0_220_1M.txt",
        "induct_train": f"/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/std_exps_1M_lime_run1/lime_induct/train1000000_deduped_lime_induct0_220_1M.txt",
        "induct_valid": f"/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/std_exps_1M_lime_run1/lime_induct/valid10000_deduped_lime_induct0_220_1M.txt"}

vm3_nesting_language_base_dir = '/mnt/disks/persist/felix-artificial-language/build_t5x_data/data/dependency_nesting_language/32000'
vm3_nesting_language_paths = [osp.join(vm3_nesting_language_base_dir, 'train.txt'), osp.join(vm3_nesting_language_base_dir, 'valid.txt')]

PRETRAIN_TASK_ID_TO_PATHS = {
  'lime': list(vm6_lime_paths.values()),
  'nonsense_summary': [f'{nonsense_summary_base}/train.txt', f'{nonsense_summary_base}/val.txt'],
  'nesting_language': vm3_nesting_language_paths,
}


simpler_tasks = ['set', 'copy', 'delete', 'sort', 'union', 'set_1_minus_2', 'set_2_minus_1', 'replace', 'duplicate', 'intersect', 'reverse', \
    'deduplicate', 'last_char', 'first_char', 'search', 'longest_word', 'length', 'count']
def fill_pretrain_task_id_to_path_dic_with_simpler_tasks():
  unary = ['copy', 'deduplicate',  'duplicate',  'first_char',  'last_char',  'length',  'longest_word',  'reverse',  'set']
  binary = ['search', 'set_1_minus_2',  'set_2_minus_1',  'sort' , 'union', 'replace' ,'delete', 'count', 'intersect']

  assert len(simpler_tasks) == 18
  assert set(unary + binary) == set(simpler_tasks)

  unary_base = '/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/1M_1_23_unary'
  for t in unary:
    train, valid = (osp.join(unary_base, t, x) for x in [f'train1000000_deduped_{t}10_220_1M.txt',  f'valid10000_deduped_{t}10_220_1M.txt'])
    assert osp.isfile(valid) and osp.isfile(train)
    PRETRAIN_TASK_ID_TO_PATHS[t] = [train, valid]
  binary_base = '/mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/1M_1_25_binary'
  for t in binary:
    train, valid = (osp.join(binary_base, t, x) for x in [f'train1000000_deduped_{t}10_220_1M.txt',  f'valid10000_deduped_{t}10_220_1M.txt'])
    assert osp.isfile(valid) and osp.isfile(train)
    PRETRAIN_TASK_ID_TO_PATHS[t] = [train, valid]
  
fill_pretrain_task_id_to_path_dic_with_simpler_tasks()

def copy_simpler_pretrain_tasks_to_gcs():
  for t in simpler_tasks:
    train, valid = PRETRAIN_TASK_ID_TO_PATHS[t]
    gcs_base = f'gs://n2formal-public-data/synthetic_pretraining/pretraining_data/simpler_tasks/{t}/'
    train_gcs_path = osp.join(gcs_base, 'train.txt')
    valid_gcs_path = osp.join(gcs_base, 'valid.txt')
    bash(f'gsutil cp {train} {train_gcs_path}')
    bash(f'gsutil cp {valid} {valid_gcs_path}')
    # break

# copy_simpler_pretrain_tasks_to_gcs()

def copy_downstream_tasks_to_gcs():
  for t, paths in DOWNSTREAM_TASK_ID_TO_PATHS.items():
    for p in paths:
      print(p)
      assert osp.isfile(p)
    train, valid, test = paths
    gcs_base = f'gs://n2formal-public-data/synthetic_pretraining/finetuning_data/{t}/'
    train_gcs_path = osp.join(gcs_base, 'train.txt')
    valid_gcs_path = osp.join(gcs_base, 'valid.txt')
    test_gcs_path = osp.join(gcs_base, 'test.txt')
    bash(f'gsutil cp {train} {train_gcs_path}')
    bash(f'gsutil cp {valid} {valid_gcs_path}')
    bash(f'gsutil cp {test} {test_gcs_path}')

# copy_downstream_tasks_to_gcs()


# def copy_lime_to_gcs():
#   gcs_base = f'gs://n2formal-public-data/synthetic_pretraining/pretraining_data/lime/'

#   for k, v in vm6_lime_paths.items():
#     print(v)
#     assert osp.isfile(v)
#     gcs_path = osp.join(gcs_base, f'{k}.txt')
#     print(gcs_path)
#     bash(f'gsutil cp {v} {gcs_path}')

# copy_lime_to_gcs()


def copy_task_to_gcs(task):
  assert task in ['nonsense_summary', 'nesting_language']
  gcs_base = f'gs://n2formal-public-data/synthetic_pretraining/pretraining_data/{task}/'

  train, valid = PRETRAIN_TASK_ID_TO_PATHS[task]
  assert osp.isfile(train)
  assert osp.isfile(valid)

  train_gcs_path = osp.join(gcs_base, f'train.txt')
  valid_gcs_path = osp.join(gcs_base, f'valid.txt')
  bash(f'gsutil cp {train} {train_gcs_path}')
  bash(f'gsutil cp {valid} {valid_gcs_path}')
# copy_task_to_gcs('nesting_language')
# copy_task_to_gcs('nonsense_summary')
# modified_text-to-text-transfer-transformer