from os import system as bash
import os.path as osp


def get_ckpt_step(vocab_size):
  # return step that pretraining hits max validation accuracy (for copy pretrain, max validation acc is 1.0)
  if vocab_size == 2:
    step = 30000
  elif vocab_size == 1000:
    step = 20000
  elif vocab_size == 10000:
    step = 20000
  else:
    raise NotImplementedError
  return step


def get_ckpt_path(vocab_size):
  step = get_ckpt_step(vocab_size)
  return f'/mnt/disks/persist/t5_training_models/final_exps/mar25_copy_ablate_vocab/pretrain_{vocab_size}/checkpoint_{step}'


def make_finetune_command(task, vocab_size, data_min_length, data_max_length, vm, bash_arg4, override_exp_name=None):
  ckpt = get_ckpt_path(vocab_size)
  step = get_ckpt_step(vocab_size)
  assert str(step) in ckpt
  exp_name = osp.join('final_exps/mar25_copy_vocab_ablations/', f'{task}_vocabsize{vocab_size}_length{data_min_length}_{data_max_length}_ckpt{step}')
  if override_exp_name is not None:
    exp_name = override_exp_name
  command = f'bash final_exps/mar25_copy_ablate_vocab/{task}/finetune_ARGS_checkpoint_exp_vm_idk.sh {ckpt} {exp_name} {vm} {bash_arg4}'
  return command


def load_all_except_embed_make_finetune_command(task, vocab_size, data_min_length, data_max_length, vm, bash_arg4, override_exp_name=None):
  ckpt = get_ckpt_path(vocab_size)
  step = get_ckpt_step(vocab_size)
  assert str(step) in ckpt
  exp_name = osp.join('final_exps/mar25_copy_vocab_ablations/', f'loadallexceptembed_{task}_vocabsize{vocab_size}_length{data_min_length}_{data_max_length}_ckpt{step}')
  if override_exp_name is not None:
    exp_name = override_exp_name
  command = f'bash final_exps/mar25_copy_ablate_vocab/{task}/load_all_except_embed_finetune.sh {ckpt} {exp_name} {vm} {bash_arg4}'
  return command


if __name__ == '__main__':
  print(make_finetune_command('cnndm', 1000, 2, 1, 'tmp', -1))
