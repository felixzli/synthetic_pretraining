import os.path as osp
import sys
import os
sys.path.append('final_exps/final_exps_utils/ckpts/pretrain_ckpts/')
sys.path.append(os.getcwd())

import cfg_t5_small_ckpts 
import feb13_t5_small_ckpts 
import language_t5_small_ckpts 
import copy_ablations_t5_small_ckpts
import artificial_language_ckpts
from cfg_t5_small_ckpts import PRETRAIN_IDS as cfg_pretrain_ids
from feb13_t5_small_ckpts import PRETRAIN_IDS as feb13_pretrain_ids
from language_t5_small_ckpts import PRETRAIN_IDS as language_pretrain_ids
from copy_ablations_t5_small_ckpts import PRETRAIN_IDS as copy_ablations_pretrain_ids
from artificial_language_ckpts import PRETRAIN_IDS as artificial_language_pretrain_ids



PRETRAIN_IDS = set(list(cfg_pretrain_ids) + list(feb13_pretrain_ids) + list(language_pretrain_ids) + list(copy_ablations_pretrain_ids) + list(artificial_language_pretrain_ids))
assert len(PRETRAIN_IDS) == len(cfg_pretrain_ids) + len(feb13_pretrain_ids) + len(language_pretrain_ids) + len(copy_ablations_pretrain_ids) + len(artificial_language_pretrain_ids)


def get_ckpt_path_given_pretrain_id(pretrain_id):
  if pretrain_id == 'from_scratch':
    return pretrain_id

  if pretrain_id in cfg_pretrain_ids:
    ckpt_module = cfg_t5_small_ckpts
  elif pretrain_id in feb13_pretrain_ids:
    ckpt_module = feb13_t5_small_ckpts
  elif pretrain_id in language_pretrain_ids:
    ckpt_module = language_t5_small_ckpts
  elif pretrain_id in copy_ablations_pretrain_ids:
    ckpt_module = copy_ablations_t5_small_ckpts
  elif pretrain_id in artificial_language_pretrain_ids:
    ckpt_module = artificial_language_ckpts
  else:
    raise NotImplementedError

  ckpt_path = ckpt_module.get_local_ckpt_path(pretrain_id)
  if not osp.isdir(ckpt_path):
    print(ckpt_module)
    ckpt_module.copy_ckpt_from_gcs_to_local(pretrain_id)
    print(ckpt_module)

  assert osp.isdir(ckpt_path)
  return ckpt_path

