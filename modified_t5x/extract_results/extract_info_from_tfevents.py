import os.path as osp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import subprocess
import numpy as np
import tensorflow as tf


def bash(command):
  stdout = subprocess.check_output(command, shell=True)
  return stdout


def _convert_tf_protos_to_floats(tf_protoss):
  np_floatss = []
  for tf_proto in tf_protoss:
    np_floatss.append(float(tf.make_ndarray(tf_proto)))
  return np_floatss


def get_accs_and_steps(tfevents_path, print_result=False, print_all_accs_and_step_nums=False):
  event_acc = EventAccumulator(osp.dirname(tfevents_path),  size_guidance={'tensors': 0})
  event_acc.Reload()
  print(event_acc.Tags())
  # print(event_acc.Tags()) # through this print, can see the names of all logged values
  _, step_nums, accs = zip(*event_acc.Tensors('accuracy'))
  accs = _convert_tf_protos_to_floats(accs)
  # breakpoint()
  if len(accs) == 0:
    return -1, -1
  return accs, step_nums


def get_max_acc_and_step_num(tfevents_path, print_result=False, print_all_accs_and_step_nums=False, also_return_accs_steps=False):
  event_acc = EventAccumulator(osp.dirname(tfevents_path),  size_guidance={'tensors': 0})
  event_acc.Reload()
  # print(event_acc.Tags())
  # print(event_acc.Tags()) # through this print, can see the names of all logged values
  _, step_nums, accs = zip(*event_acc.Tensors('accuracy'))
  accs = _convert_tf_protos_to_floats(accs)
  # breakpoint()
  if len(accs) == 0:
    return -1, -1
  max_acc_idx = np.argmax(accs)
  max_acc, step_num = accs[max_acc_idx], step_nums[max_acc_idx]
  if print_all_accs_and_step_nums:
    for s, a in zip(step_nums, accs):
      print(f'{s} | {a}')

  if print_result:
    print(tfevents_path)
    print(f'max_acc: {max_acc}\nstep_num: {step_num}')

  if also_return_accs_steps:
    return max_acc, step_num, accs, step_nums
  return max_acc, step_num



def get_first_99_acc_and_step_num(tfevents_path, print_result=False, print_all_accs_and_step_nums=False):
  event_acc = EventAccumulator(osp.dirname(tfevents_path),  size_guidance={'tensors': 0})
  event_acc.Reload()
  # print(event_acc.Tags())
  # print(event_acc.Tags()) # through this print, can see the names of all logged values
  _, step_nums, accs = zip(*event_acc.Tensors('accuracy'))
  accs = _convert_tf_protos_to_floats(accs)
  # breakpoint()
  if len(accs) == 0:
    return -1, -1
  max_acc_idx = np.argmax(accs)
  max_acc, step_num = accs[max_acc_idx], step_nums[max_acc_idx]
  if print_all_accs_and_step_nums:
    for s, a in zip(step_nums, accs):
      print(f'{s} | {a}')


  for s, a in zip(step_nums, accs):
    if a >= 0.99:
      if print_result:
        print(tfevents_path)
        print(f'first_99_acc_step: {s}\nacc: {a}')
      return a, s


  if print_result:
    print(tfevents_path)
    print(f'max_acc: {max_acc}\nstep_num: {step_num}')

  return max_acc, step_num


if __name__ == '__main__':
  # tfevents_path = '/Users/felix/Documents/research/results_and_analysis/universal_lime/final_exps/mar20_cfg/deduct_pretrain_t5small/valid/mar20_cfg_deduct/events.out.tfevents.1647906168.t1v-n-fd8d27d0-w-0.2454531.1.v2'
  # print(get_max_acc_and_step_num(tfevents_path, print_result=True))
  print('.')



