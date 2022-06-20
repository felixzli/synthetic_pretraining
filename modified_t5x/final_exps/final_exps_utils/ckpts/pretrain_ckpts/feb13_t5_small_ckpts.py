import os.path as osp
from os import system as bash
from path_utils import parent_dir
import sys
import time


GCS_CKPT_BASE_DIR = osp.join('gs://n2formal-community-external-lime/universal_lime/t5x_checkpoints/', osp.basename(__file__)[:-3])
LOCAL_CKPT_BASE_DIR = osp.join('/mnt/disks/persist/t5_training_models/CHECKPOINTS_FROM_GCS/', osp.basename(__file__)[:-3])


PRETRAIN_IDS_TO_OG_CKPT_PATH = \
{
  'copy': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_single_task/unary/copy/checkpoint_5000', # vm2
  'copy_20k': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_single_task/unary/copy/checkpoint_20000', # vm2

  'set': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_single_task/unary/set/checkpoint_10000',

  'lime': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_single_task/lime/all_lime/checkpoint_80000/',
  'lime_and_unary': '/mnt/disks/persist/t5_training_models/CHECKPOINTS_FROM_GCS/t5_small/lime_and_unary/checkpoint_70000',
  '27tasks': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_mixtures/lime_AND_unary_AND_binary_AND_mbpp/checkpoint_65000/',
  'wiki_80k': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_single_task/wiki/wiki40b/checkpoint_80000', #vm5
  'wiki_30k': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_single_task/wiki/wiki40b/checkpoint_30000', #vm5
  'wiki_10k': '/mnt/disks/persist/t5_training_models/synthetic_tasks/2-13_pretraining_and_finetune_sweep/small_pretrain_single_task/wiki/wiki40b/checkpoint_10000', #vm5


  'lime_depth36':'/mnt/disks/persist/t5_training_models/final_exps/pretraining/LIME/t5_36_bs64/checkpoint_60000', #vm3
  'nonsense_summary_5k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_5000/',#vm2
  'nonsense_summary_75k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_75000/',#vm2
  'nonsense_summary_10k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_10000/',
  'nonsense_summary_20k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_20000/',
  'nonsense_summary_30k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_30000/',
  'nonsense_summary_40k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_40000/',
  'nonsense_summary_50k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_50000/',
  'nonsense_summary_60k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_60000/',
  'nonsense_summary_70k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small/checkpoint_70000/',

  'nonsense_summary_2k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small_period_500/checkpoint_2000/',

  'nonsense_summary_2_5k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small_period_500/checkpoint_2500/',
  'nonsense_summary_3k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small_period_500/checkpoint_3000/',
  'nonsense_summary_4k': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/nonsense_summary/t5_small_period_500/checkpoint_4000/',


  't511_offshelf':'/mnt/disks/persist/t5_training_models/offshelf_t511_small/checkpoint_1000001',

  'std_exps_lime_run1': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/LIME/std_exps/run1/checkpoint_70000', #vm6
  'std_exps_lime_run2': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/LIME/std_exps/run2/checkpoint_45000',
  'std_exps_lime_run3': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/LIME/std_exps/run3/checkpoint_30000',
  'std_exps_lime_run4': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/LIME/std_exps/run4/checkpoint_30000',

  'std_exps_set_run1': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/set/std_exps/run1/checkpoint_10000', #vm6
  'std_exps_set_run2': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/set/std_exps/run2/checkpoint_10000',
  'std_exps_set_run3': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/set/std_exps/run3/checkpoint_10000',
  'std_exps_set_run4': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/set/std_exps/run4/checkpoint_10000',


  'std_exps_copy_run1': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/copy/std_exps/run1/checkpoint_5000', #vm2
  'std_exps_copy_run2': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/copy/std_exps/run2/checkpoint_5000',
  'std_exps_copy_run3': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/copy/std_exps/run3/checkpoint_5000',
  'std_exps_copy_run4': '/mnt/disks/persist/t5_training_models/final_exps/pretraining/copy/std_exps/run4/checkpoint_5000',

}

PRETRAIN_IDS = set(PRETRAIN_IDS_TO_OG_CKPT_PATH.keys())


def get_local_ckpt_path(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  return osp.join(LOCAL_CKPT_BASE_DIR, pretrain_id, 'checkpoint/')


def get_gcs_ckpt_path(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  return osp.join(GCS_CKPT_BASE_DIR, pretrain_id, 'checkpoint/')


def get_original_ckpt_path(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  return PRETRAIN_IDS_TO_OG_CKPT_PATH[pretrain_id]


def copy_ckpt_to_gcs(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  
  __pretrain_exp_ckpt_path = get_original_ckpt_path(pretrain_id)
  
  pretrain_exp_ckpt_path = osp.join(parent_dir(__pretrain_exp_ckpt_path), 'checkpoint/')
  bash(f'rm -rf {pretrain_exp_ckpt_path}')
  assert len(pretrain_exp_ckpt_path) < len(__pretrain_exp_ckpt_path)
  bash(f'cp -r {__pretrain_exp_ckpt_path} {pretrain_exp_ckpt_path}')
  gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)
  bash(f'gsutil -m rm -rf {parent_dir(gcs_ckpt_path)}')
  print(pretrain_exp_ckpt_path)
  print(gcs_ckpt_path)
  command = f'gsutil -m cp -r {pretrain_exp_ckpt_path} {parent_dir(gcs_ckpt_path)}/'

  print(command)
  bash(command)
  print('- finished copy_ckpt_to_gcs -')


def copy_ckpt_from_gcs_to_local(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  local_ckpt_path = get_local_ckpt_path(pretrain_id)
  gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)

  bash(f'rm -rf {parent_dir(local_ckpt_path)}')
  bash(f'mkdir -p {parent_dir(local_ckpt_path)}')
  command = f'gsutil -m cp -r {gcs_ckpt_path} {parent_dir(local_ckpt_path)}'

  print(command)
  bash(command)
  print('- finished copy_ckpt_from_gcs_to_local -')



if __name__ == '__main__':
  pretrain_id = sys.argv[1]
  print(PRETRAIN_IDS)
  print(pretrain_id)
  if len(sys.argv) > 2 and (sys.argv[2] == 'copy_ckpt_to_gcs' or sys.argv[2] == 'cctg'):
    copy_ckpt_to_gcs(pretrain_id)
    time.sleep(0.5)
  copy_ckpt_from_gcs_to_local(pretrain_id)
  time.sleep(0.5)
  print(get_local_ckpt_path(pretrain_id))
  bash(f'ls {get_local_ckpt_path(pretrain_id)}')
  print(get_local_ckpt_path(pretrain_id))

  


# python final_exps/final_exps_utils/ckpts/pretrain_ckpts/feb13_t5_small_ckpts.py std_exps_copy_run1 cctg
# python final_exps/final_exps_utils/ckpts/pretrain_ckpts/feb13_t5_small_ckpts.py std_exps_copy_run2 cctg
# python final_exps/final_exps_utils/ckpts/pretrain_ckpts/feb13_t5_small_ckpts.py std_exps_copy_run3 cctg
# python final_exps/final_exps_utils/ckpts/pretrain_ckpts/feb13_t5_small_ckpts.py std_exps_copy_run4 cctg