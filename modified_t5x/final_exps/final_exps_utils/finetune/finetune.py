import os
import os.path as osp
from os import system as bash
import sys
sys.path.append(os.getcwd())
from path_utils import parent_dir
import argparse
from final_exps.final_exps_utils.ckpts.get_ckpt_path_given_pretrain_id import get_ckpt_path_given_pretrain_id
import time




TASKS = ['cnndm', 'isarstep', 'mar30_codetrans', 'apr2_retrosynthesis', 'apr3_homology', 'sst2', 'mtop']

def make_command(finetune_bash_file, ckpt_path, exp, vm, idk, init_id=None):
  if init_id is None:
    return f'bash {finetune_bash_file} {ckpt_path} {exp} {vm} {idk}'
  else:
    return f'bash {finetune_bash_file} {ckpt_path} {exp} {vm} {idk} {init_id}'


def _make_relpos_grid_search_command(finetune_bash_file, ckpt_path, exp, vm, idk, init_with_custom_std):
  return f'bash {finetune_bash_file} {ckpt_path} {exp} {vm} {idk} {init_with_custom_std}'


def _make_pre_attn_ln_value_sweep_finetune_command(finetune_bash_file, ckpt_path, exp, vm, idk, init_with_custom_std):
  return f'bash {finetune_bash_file} {ckpt_path} {exp} {vm} {idk} {init_with_custom_std}'


def str_to_bool(value):
  if isinstance(value, bool):
      return value
  if value.lower() in {'false', 'f', '0', 'no', 'n'}:
      return False
  elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
      return True
  raise ValueError(f'{value} is not a valid boolean value')


def run_relpos_std_sweep_finetune_command(finetune_bash_file, vm, idk, enc_relpos_std, dec_relpos_std):
  init_with_custom_std = f'enc_relpos-{enc_relpos_std}---dec_relpos-{dec_relpos_std}'
  command = f"python final_exps/final_exps_utils/finetune/finetune.py --vm {vm} --idk {idk} --finetune_bash_file {finetune_bash_file} --init_with_custom_std {init_with_custom_std} --is_relpos_std_grid_search True"
  os.system(command)


def run_pre_attn_ln_value_sweep_finetune_command(finetune_bash_file, vm, idk, value):
  if finetune_bash_file == 'final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh':
    init_with_custom_value = f'pre_self_attn_ln-{value}---pre_cross_attn_ln-{value}---pre_mlp_ln-{value}'
    print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
    print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
    print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
    print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
    print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")

  else:
    init_with_custom_value = f'pre_self_attn_ln-{value}---pre_cross_attn_ln-{value}'
  command = f"python final_exps/final_exps_utils/finetune/finetune.py --vm {vm} --idk {idk} --finetune_bash_file {finetune_bash_file} --init_with_custom_value {init_with_custom_value} --is_pre_attn_ln_value_sweep True"
  os.system(command)


def run_pre_attn_ln_and_pre_mlp_ln_value_sweep_finetune_command(finetune_bash_file, vm, idk, value):
  init_with_custom_value = f'pre_self_attn_ln-{value}---pre_cross_attn_ln-{value}---pre_mlp_ln-{value}'
  print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
  print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
  print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
  print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")
  print("final_exps/language_tasks/cnndm/runs/pre_attn_ln_value_sweep/t5_small/_set_pre_attn_ln_and_pre_mlp_ln_value.sh")

  command = f"python final_exps/final_exps_utils/finetune/finetune.py --vm {vm} --idk {idk} --finetune_bash_file {finetune_bash_file} --init_with_custom_value {init_with_custom_value} --is_pre_attn_ln_value_sweep True"
  os.system(command)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_suffix", type=str, required=False, default=None)
  parser.add_argument("--pretrain_id", type=str, required=False, default=None)
  parser.add_argument("--vm", type=str, required=True)
  parser.add_argument("--idk", type=str, default='xxxxxxxxxx', required=False)
  parser.add_argument("--init_id", type=str, default=None, required=False)

  
  parser.add_argument("--finetune_bash_file", type=str, required=True)

  parser.add_argument("--is_relpos_std_grid_search", type=str_to_bool, nargs='?', const=False, default=False)
  parser.add_argument("--init_with_custom_std", type=str, default=None)


  parser.add_argument("--is_pre_attn_ln_value_sweep", type=str_to_bool, nargs='?', const=False, default=False)
  parser.add_argument("--init_with_custom_value", type=str, default=None)
  # enc_relpos-8.7---dec_relpos-4

  args = parser.parse_args()

  exp_suffix = args.exp_suffix
  pretrain_id = args.pretrain_id
  vm = args.vm
  idk = args.idk
  init_id = args.init_id
  finetune_bash_file = args.finetune_bash_file
  is_relpos_std_grid_search = args.is_relpos_std_grid_search
  is_pre_attn_ln_value_sweep = args.is_pre_attn_ln_value_sweep

  if is_relpos_std_grid_search:
    init_with_custom_std = args.init_with_custom_std
    assert init_with_custom_std is not None 

    assert osp.isfile(finetune_bash_file)
    pretrain_id = 'from_scratch'
    if idk != 'scp_results_no_preds':
      ckpt_path = get_ckpt_path_given_pretrain_id(pretrain_id)
    else:
      ckpt_path = 'xxx'

    exp_suffix = init_with_custom_std
    exp = osp.join(parent_dir(finetune_bash_file), 'relpos_std_grid_search', exp_suffix)
    command = _make_relpos_grid_search_command(finetune_bash_file, ckpt_path, exp, vm, idk, init_with_custom_std)
    print('=====================================')
    print(f'===== EXP ==== {exp}')
    print(f'===== CKPT_PATH ===== {ckpt_path}')
    print(f'===== INIT ===== {init_with_custom_std}')
    print(f'===== COMMAND ===== {command}')


    time.sleep(2)
    bash(command)
  elif is_pre_attn_ln_value_sweep:
    init_with_custom_value = args.init_with_custom_value
    assert init_with_custom_value is not None 

    assert osp.isfile(finetune_bash_file)
    pretrain_id = 'from_scratch'
    if idk != 'scp_results_no_preds':
      ckpt_path = get_ckpt_path_given_pretrain_id(pretrain_id)
    else:
      ckpt_path = 'xxx'

    exp = osp.join(parent_dir(finetune_bash_file), 'pre_attn_ln_value_sweep', init_with_custom_value)
    command = _make_pre_attn_ln_value_sweep_finetune_command(finetune_bash_file, ckpt_path, exp, vm, idk, init_with_custom_value)
    print('=====================================')
    print(f'===== EXP ==== {exp}')
    print(f'===== CKPT_PATH ===== {ckpt_path}')
    print(f'===== INIT ===== {init_with_custom_value}')
    print(f'===== COMMAND ===== {command}')

    if idk != 'norun':
      time.sleep(2)
      bash(command)

  else:
    assert exp_suffix is not None
    assert pretrain_id is not None
    # print('\n----')
    # print(finetune_bash_file)
    assert osp.isfile(finetune_bash_file)
    if idk != 'scp_results_no_preds':
      ckpt_path = get_ckpt_path_given_pretrain_id(pretrain_id)
    else:
      ckpt_path = 'xxx'
    if init_id is not None and idk=='cid':
      try:
        sys.path.append('./t5x')
        grouping, stats_to_init = init_id.split('___')
        assert '___' in init_id
        exec(f'from t5x._final_init_exps import get_{grouping}')
        exec(f'from t5x._final_init_exps import {stats_to_init}')
        print("VALID INIT ID")
      except:
        print(init_id)
        print("FAIL FAIL FAIL FAIL INVALID INIT ID !()@#*!)@(*#)(!@*#)(!@*)#(!@*#()!*@)(#*!@)(#*@()#*!)@(*#)(!@*#)(!@*#)(!@!@*(#&!@(*#&!@(*#&*(!@&#(*!@&$)@*)$!(@*$)(!@*KJQSDNKJN*")
      exit()

    exp = osp.join(parent_dir(finetune_bash_file), pretrain_id, exp_suffix)
    command = make_command(finetune_bash_file, ckpt_path, exp, vm, idk, init_id=init_id)
    print(f'===== COMMAND ===== {command}')
    print(f'===== EXP ==== {exp}')
    print(f'===== CKPT_PATH ===== {ckpt_path}')
    if idk != 'norun':
      time.sleep(2)
      bash(command)