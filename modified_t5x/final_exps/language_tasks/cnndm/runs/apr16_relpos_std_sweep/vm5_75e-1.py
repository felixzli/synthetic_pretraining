# enc_relpos-8.7---dec_relpos-4


import sys
import os.path as osp
import os
sys.path.append(os.getcwd())
from final_exps.final_exps_utils.finetune.finetune import run_relpos_std_sweep_finetune_command


finetune_bash_file = 'final_exps/language_tasks/cnndm/_relpos_std_sweep.sh'
vm=5

enc_relpos_std = 7.5
dec_relpos_stds = [0.2, 2.5, 5.0, 7.5, 10.0]
idk=sys.argv[1]

# def run_relpos_std_sweep_finetune_command(finetune_bash_file, vm, idk, enc_relpos_std, dec_relpos_std):

for dec_relpos_std in dec_relpos_stds:
  run_relpos_std_sweep_finetune_command(finetune_bash_file, vm, idk, enc_relpos_std, dec_relpos_std)