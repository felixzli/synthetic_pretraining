import os.path as osp
import seqio
import sys
sys.path.append(osp.dirname(__file__))
from add_tasks_utils import translation_processors, t5_output_features, DEFAULT_OUTPUT_FEATURES, DATA_BASE_DIR
from t5.evaluation import metrics, qa_utils
from t5.data import postprocessors
import inspect


custom_downstream_tasks = ['cnndm_10k','code_translation', 'mtop', 'retrosynthesis', 'webqsp']


def add_custom_downstream_task(task_registry, task_id):
  # print(task_id)
  assert task_id in custom_downstream_tasks
  base = osp.join(DATA_BASE_DIR, 'finetuning_data', task_id)
  split = {
    'train': osp.join(base, 'train.txt'),
    'validation': osp.join(base, 'valid.txt'),
    'test': osp.join(base, 'test.txt')
  }
  for v in split.values():
    # print(v)
    assert osp.isfile(v)
    
  if task_id == 'cnndm_10k':
    metric = metrics.rouge
  else:
    metric = qa_utils.semantic_parsing_metrics
  task_registry.add(
    task_id,
    source=seqio.TextLineDataSource(split),
    output_features=t5_output_features("t5"),
    preprocessors=translation_processors(),
    metric_fns=[metric])












# def add_code_translation(task_registry):
#   task_id = get_task_id_from_function_name(get_current_function_name())
#   # print(task_id)
#   assert task_id in custom_downstream_tasks
#   base = osp.join(DATA_BASE_DIR, 'finetuning_data', task_id)
#   split = {
#     'train': osp.join(base, 'train.txt'),
#     'validation': osp.join(base, 'valid.txt'),
#     'test': osp.join(base, 'test.txt')
#   }
#   for v in split.values():
#     # print(v)
#     assert osp.isfile(v)

#   task_registry.add(
#     task_id,
#     source=seqio.TextLineDataSource(split),
#     output_features=t5_output_features("t5"),
#     preprocessors=translation_processors(),
#     metric_fns=[qa_utils.semantic_parsing_metrics])

# add_code_translation()

# def add_retrosynthesis(task_registry):
#   task_id = get_task_id_from_function_name(get_current_function_name())
#   # print(task_id)
#   assert task_id in custom_downstream_tasks
#   base = osp.join(DATA_BASE_DIR, 'finetuning_data', task_id)
#   split = {
#     'train': osp.join(base, 'train.txt'),
#     'validation': osp.join(base, 'valid.txt'),
#     'test': osp.join(base, 'test.txt')
#   }
#   for v in split.values():
#     # print(v)
#     assert osp.isfile(v)

#   task_registry.add(
#     task_id,
#     source=seqio.TextLineDataSource(split),
#     output_features=t5_output_features("t5"),
#     preprocessors=translation_processors(),
#     metric_fns=[qa_utils.semantic_parsing_metrics])

