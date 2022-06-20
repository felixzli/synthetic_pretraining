
import sys

from torch import exp_
from utils import get_steps_and_metrics
import os.path as osp
from extract_info_from_tfevents import get_max_acc_and_step_num
import glob
import numpy as np


def get_inference_json_path(exp_logdir, step):
  return glob.glob(osp.join(exp_logdir, 'inference_eval', '*metrics.jsonl'))[0]


def get_max_val_acc_metric(exp_logdir, metric_namess, print_result=False, print_all_accs_and_step_nums=False):
  '''
  return: (step, max_val_acc, dictionary of metric name to metric value)
  '''


  if not osp.isdir(exp_logdir) and 'STD' in exp_logdir:
    exp_logdir = exp_logdir.replace('STD', 'VAR')

  if not osp.isdir(exp_logdir) and 'VAR' in exp_logdir:
    exp_logdir = exp_logdir.replace('VAR', 'STD')


  print(exp_logdir)

  assert osp.isdir(exp_logdir)
  valid_tfevents_pathss = glob.glob(osp.join(exp_logdir, 'training_eval', '*', 'events*'))
  assert len(valid_tfevents_pathss) > 0
  if len(valid_tfevents_pathss) > 1:
    # raise NotImplementedError('have not implemented how to handle multiple tfevents (when they exist due to resuming finetuning)')
    max_val_acc = -1
    max_val_acc_step = -1
    accs = []
    steps = []
    for path in valid_tfevents_pathss:
      try:
        path_max_val_acc, path_max_val_acc_step, path_accs, path_steps = get_max_acc_and_step_num(path, print_all_accs_and_step_nums=print_all_accs_and_step_nums, also_return_accs_steps=True)
        if path_max_val_acc > max_val_acc:
          max_val_acc = path_max_val_acc
          max_val_acc_step = path_max_val_acc_step
          accs.append(path_accs)
          steps.append(path_steps)
      except:
        continue
  else:
    valid_tf_event_path = valid_tfevents_pathss[0]
    max_val_acc, max_val_acc_step, accs, steps = get_max_acc_and_step_num(valid_tf_event_path, print_all_accs_and_step_nums=print_all_accs_and_step_nums, also_return_accs_steps=True)

  inference_json_path = get_inference_json_path(exp_logdir, max_val_acc_step)
  stepss, metricss = get_steps_and_metrics(inference_json_path)

  print(f'steps: {steps}')
  print(f'accs: {accs}')
  print(max_val_acc_step)
  try:
    max_val_acc_step_idx = np.where(np.array(stepss)==max_val_acc_step)[0][0]
    assert stepss[max_val_acc_step_idx] == max_val_acc_step
    max_val_acc_metric_dic = {k:metricss[k][max_val_acc_step_idx] for k in metric_namess}
  except:
    max_val_acc_metric_dic = 'max val acc metric not logged'
  max_metric_dic = {k:max(metricss[k]) for k in metric_namess}

  for n in metric_namess:
    print(f'{n}: {metricss[n]}')
  if print_result:
    print('--'*10)
    print(f'=========={exp_logdir}')
    print(f'step: {max_val_acc_step}')
    print(f'val acc: {max_val_acc}')
    print(f'max_val_acc metrics: {max_val_acc_metric_dic}')
    print(f'max metrics: {max_metric_dic}')
  return max_val_acc_step, max_val_acc, max_val_acc_metric_dic

if __name__ == '__main__':
  exp_logdir = sys.argv[1]
  metrics = sys.argv[2]
  metrics = metrics.split(',')
  if len(sys.argv) == 3:
    get_max_val_acc_metric(exp_logdir, metrics, print_result=True)
  else:
    get_max_val_acc_metric(exp_logdir, metrics, print_result=True, print_all_accs_and_step_nums=True)
