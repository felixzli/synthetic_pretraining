import os.path as osp
from os import system as bash
from path_utils import parent_dir
import sys
import time
import sys
import os
sys.path.append(os.getcwd())
from final_exps.final_exps_utils.ckpts.copy_original_ckpt_to_gcs import copy_original_ckpt_to_gcs
from final_exps.final_exps_utils.ckpts.copy_ckpt_from_gcs_to_local import copy_ckpt_from_gcs_to_local as ccfgtl


GCS_CKPT_BASE_DIR = osp.join('gs://n2formal-community-external-lime/universal_lime/t5x_checkpoints/', osp.basename(__file__)[:-3])
LOCAL_CKPT_BASE_DIR = osp.join('/mnt/disks/persist/t5_training_models/CHECKPOINTS_FROM_GCS/', osp.basename(__file__)[:-3])


PRETRAIN_IDS_TO_OG_CKPT_PATH = \
{
  # vm3
  'nesting_dependency_language':'/mnt/disks/persist/t5_training_models/final_exps/pretraining/artificial_language/nesting_dependency_language/apr9_first_try/t5_small/checkpoint_170000',
  'default_t5_task_params_nesting_dependency_language':'/mnt/disks/persist/t5_training_models/final_exps/pretraining/artificial_language/nesting_dependency_language/apr10_default_t5_task_params/t5_small/checkpoint_165000', 
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


def copy_original_ckpt_to_gcs_given_pretrain_id(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  
  original_ckpt_path = get_original_ckpt_path(pretrain_id)
  gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)

  copy_original_ckpt_to_gcs(original_ckpt_path, gcs_ckpt_path)


def copy_ckpt_from_gcs_to_local(pretrain_id):
  assert pretrain_id in PRETRAIN_IDS
  gcs_ckpt_path = get_gcs_ckpt_path(pretrain_id)
  local_ckpt_path = get_local_ckpt_path(pretrain_id)
  ccfgtl(gcs_ckpt_path, local_ckpt_path)


if __name__ == '__main__':
  if sys.argv[1] == 'check':
    for pid in PRETRAIN_IDS:
      print(pid)
      assert osp.isdir(get_local_ckpt_path(pid))
    exit()
  pretrain_id = sys.argv[1]


  if len(sys.argv) > 2 and sys.argv[2] == 'copy_ckpt_to_gcs':
    print(get_original_ckpt_path(pretrain_id))

    copy_original_ckpt_to_gcs_given_pretrain_id(pretrain_id)
    time.sleep(0.5)

  copy_ckpt_from_gcs_to_local(pretrain_id)
  time.sleep(0.5)

  print(get_local_ckpt_path(pretrain_id))
  bash(f'ls {get_local_ckpt_path(pretrain_id)}')
  print(get_gcs_ckpt_path(pretrain_id))
  print(get_local_ckpt_path(pretrain_id))



  
