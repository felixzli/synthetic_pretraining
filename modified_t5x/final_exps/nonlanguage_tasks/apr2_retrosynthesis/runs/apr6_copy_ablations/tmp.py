import sys
import os
from os import system as bash
sys.path.append(os.getcwd())
from final_exps.final_exps_utils.ckpts.pretrain_ckpts.copy_ablations_t5_small_ckpts import PRETRAIN_IDS


for pid in PRETRAIN_IDS:
  command = f'cp final_exps/nonlanguage_tasks/apr2_retrosynthesis/runs/apr2_first_try/27tasks.sh final_exps/nonlanguage_tasks/apr2_retrosynthesis/runs/apr6_copy_ablations/{pid}.sh'
  bash(command)