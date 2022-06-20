from os import system as bash
import os.path as osp


def get_ckpt_step(length):
  # return step that pretraining hits max validation accuracy (for copy pretrain, max validation acc is 1.0)
  if length == 28:
    step = 20000
  elif length == 252:
    step = 10000
  elif length == 1020:
    step = 15000
  elif length=='jan23':
    step = 30000
  else:
    raise NotImplementedError
  return step


def get_ckpt_path(length):
  step = get_ckpt_step(length)
  return f'/mnt/disks/persist/t5_training_models/final_exps/mar21_copy_length/pretrain_{length}/checkpoint_{step}'


def make_finetune_command(task, length, vm, bash_arg4, override_exp_name=None):
  ckpt = get_ckpt_path(length)
  step = get_ckpt_step(length)
  assert str(step) in ckpt
  exp_name = osp.join('final_exps/mar21_copy_length_ablations/', f'{task}_copy{length}_ckpt{step}')
  if override_exp_name is not None:
    exp_name = override_exp_name
  command = f'bash final_exps/mar21_copy_length/{task}/finetune_ARGS_checkpoint_exp_vm_idk.sh {ckpt} {exp_name} {vm} {bash_arg4}'
  return command


if __name__ == '__main__':
  print(make_finetune_command('cnndm', 1020, 2, 1, 'tmp'))
