import sys
import os
sys.path.append(os.getcwd())
from final_exps.final_exps_utils.ckpts.pretrain_ckpts.copy_ablations_t5_small_ckpts import PRETRAIN_IDS


for pid in PRETRAIN_IDS:
  if len(sys.argv) < 2:
    sys.argv.append('no_ctg')
  ctg_command = f'python final_exps/final_exps_utils/ckpts/pretrain_ckpts/copy_ablations_t5_small_ckpts.py {pid} copy_ckpt_to_gcs'
  no_ctg_command = f'python final_exps/final_exps_utils/ckpts/pretrain_ckpts/copy_ablations_t5_small_ckpts.py {pid}'
  if sys.argv[1] == 'ctg_voc':
    if 'voc' in pid:
      os.system(ctg_command)
    else:
      os.system(no_ctg_command)
  elif sys.argv[1] == 'ctg_len':
    if 'voc' not in pid:
      os.system(ctg_command)
    else:
      os.system(no_ctg_command)
  elif sys.argv[1] == 'no_ctg':
    os.system(no_ctg_command)
  else:
    raise NotImplementedError
  

