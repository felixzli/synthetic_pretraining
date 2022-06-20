# enc_relpos-8.7---dec_relpos-4

import sys
import os.path as osp
import os
sys.path.append(os.getcwd())
from final_exps.final_exps_utils.finetune.finetune import run_pre_attn_ln_value_sweep_finetune_command
import glob
# task = osp.basename(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))

# finetune_bash_file = f'final_exps/language_tasks/{task}/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_value.sh'
finetune_bash_file = glob.glob(osp.join(osp.dirname(osp.dirname(__file__)), '*set*value*.sh'))[0]
print(finetune_bash_file)
pre_attn_ln_values=[0.1, 0.2, 0.4, 0.8, 0.05]
pre_attn_ln_values.reverse()
idk=sys.argv[1]
vm=osp.basename(__file__)[:-3]
# print(vm)


# def run_relpos_std_sweep_finetune_command(finetune_bash_file, vm, idk, enc_relpos_std, dec_relpos_std):

for pre_attn_ln_value in pre_attn_ln_values:
  run_pre_attn_ln_value_sweep_finetune_command(finetune_bash_file, vm, idk, pre_attn_ln_value)