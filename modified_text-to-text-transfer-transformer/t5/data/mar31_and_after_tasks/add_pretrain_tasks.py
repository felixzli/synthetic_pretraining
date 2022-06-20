import os.path as osp
import seqio
import sys
sys.path.append(osp.dirname(__file__))
from add_tasks_utils import translation_processors, t5_output_features, artificial_language_processors, lime_processors, DATA_BASE_DIR
from t5.evaluation import metrics, qa_utils
import os
sys.path.append(os.getcwd())
from t5.data import preprocessors
import functools
import inspect


def get_current_function_name():
  return inspect.stack()[1][3]


def get_task_id_from_function_name(func_name):
  return '_'.join(func_name.split('_')[1:])


def add_nonsense_summary(task_registry):
  task_id = get_task_id_from_function_name(get_current_function_name())
  base = osp.join(DATA_BASE_DIR, 'pretraining_data', task_id)
  split = {
    'train': osp.join(base, 'train.txt'),
    'validation': osp.join(base, 'valid.txt'),
  }
  for v in split.values():
    # print(v)
    assert osp.isfile(v)

  task_registry.add(
    task_id,
    source=seqio.TextLineDataSource(split),
    output_features=t5_output_features("t5"),
    preprocessors=translation_processors(),
    metric_fns=[])


def add_nesting_language(task_registry):
  task_id = get_task_id_from_function_name(get_current_function_name())
  base = osp.join(DATA_BASE_DIR, 'pretraining_data', task_id)
  split = {
    'train': osp.join(base, 'train.txt'),
    'validation': osp.join(base, 'valid.txt'),
  }
  for v in split.values():
    # print(v)
    assert osp.isfile(v)

  task_registry.add(
    task_id,
    source=seqio.TextLineDataSource(split),
    output_features=t5_output_features("t5"),
    preprocessors=artificial_language_processors() + [seqio.CacheDatasetPlaceholder(),
                                                      preprocessors.span_corruption, 
                                                      seqio.preprocessors.append_eos_after_trim],
    metric_fns=[])


def add_lime(task_registry, mixture_registry):
  task_id = get_task_id_from_function_name(get_current_function_name())
  base = osp.join(DATA_BASE_DIR, 'pretraining_data', task_id)
  abduct_split = {
      "train": osp.join(base, 'abduct_train.txt'),
      "validation": osp.join(base, 'abduct_valid.txt') 
  }

  deduct_split = {
      "train": osp.join(base, 'deduct_train.txt'),
      "validation": osp.join(base, 'deduct_valid.txt') 
  }
  induct_split = {
      "train": osp.join(base, 'induct_train.txt'),
      "validation": osp.join(base, 'induct_valid.txt') 
  }

  task_registry.add(
      f"induct",
      source=seqio.TextLineDataSource(induct_split),
      output_features=t5_output_features("t5"),
      preprocessors=lime_processors(),
      metric_fns=[])


  task_registry.add(
      f"deduct",
      source=seqio.TextLineDataSource(deduct_split),
      output_features=t5_output_features("t5"),
      preprocessors=lime_processors(),
      metric_fns=[])


  task_registry.add(
      f"abduct",
      source=seqio.TextLineDataSource(abduct_split),
      output_features=t5_output_features("t5"),
      preprocessors=lime_processors(),
      metric_fns=[])


  mixture_registry.add(
    f'lime',
    [f'induct', f'deduct', f'abduct'], default_rate=1
  )



def add_simpler_tasks(task_registry):
  simpler_tasks = ['set', 'copy', 'delete', 'sort', 'union', 'set_1_minus_2', 'set_2_minus_1', 'replace', 'duplicate', 'intersect', 'reverse', \
    'deduplicate', 'last_char', 'first_char', 'search', 'longest_word', 'length', 'count']

  rename_dic = {
    'copy': 'identity',
    'set_1_minus_2':'set1_minus_set2',
    'set_2_minus_1':'set2_minus_set1',
    'first_char':'first_token',
    'last_char':'last_token'
  }
  for k in rename_dic.keys():
    assert k in simpler_tasks

  for t in simpler_tasks:
    base = osp.join(DATA_BASE_DIR, 'pretraining_data/simpler_tasks', t)
    split = {
    'train': osp.join(base, 'train.txt'),
    'validation': osp.join(base, 'valid.txt'),
    }
    for v in split.values():
      # print(v)
      assert osp.isfile(v)
    if t in rename_dic.keys():
      task_id = rename_dic[t]
    else:
      task_id = t
    
    task_registry.add(
      task_id,
      source=seqio.TextLineDataSource(split),
      output_features=t5_output_features("t5"),
      preprocessors=lime_processors(),
      metric_fns=[])