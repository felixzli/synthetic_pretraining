# enc_relpos-8.7---dec_relpos-4


import sys
import os.path as osp
import os
sys.path.append(os.getcwd())
from final_exps.final_exps_utils.finetune.finetune import run_pre_attn_ln_value_sweep_finetune_command


finetune_bash_file = 'final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_value.sh'

pre_attn_ln_values=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
idk=sys.argv[1]
vm=sys.argv[2]


# def run_relpos_std_sweep_finetune_command(finetune_bash_file, vm, idk, enc_relpos_std, dec_relpos_std):

for pre_attn_ln_value in pre_attn_ln_values:
  run_pre_attn_ln_value_sweep_finetune_command(finetune_bash_file, vm, idk, pre_attn_ln_value)