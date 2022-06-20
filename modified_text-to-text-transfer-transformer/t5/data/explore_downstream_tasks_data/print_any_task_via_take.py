import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
import seqio
from t5.data.tasks import TaskRegistry
from t5.data.mixtures import MixtureRegistry
x = os.path.dirname(os.path.dirname(__file__))
print(x)
sys.path.append(x)
from mar31_and_after_tasks.add_tasks_utils import SPM


task = sys.argv[1]
does_task_reg_ds_have_pretokenized = True if (len(sys.argv) > 2 and int(sys.argv[2]) == 1) else False

try:
  task_registry_ds = TaskRegistry.get_dataset(task, sequence_length={"inputs": 10000, "targets": 10000},
                                                    split='validation', shuffle=False)
except:
  try:
    task_registry_ds = MixtureRegistry.get_dataset(task, sequence_length={"inputs": 10000, "targets": 10000},
                                                    split='validation', shuffle=False)
  except:
    task_registry_ds = TaskRegistry.get_dataset(task, sequence_length={"inputs": 10000, "targets": 10000},
                                                    split='train', shuffle=False)

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100

if task == 'isarstep':
  VOCABULARY = SPM[task]
else:
  VOCABULARY = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


for i, data in enumerate(list(task_registry_ds.take(1))):
  # data.keys() ---> dict_keys(['inputs_pretokenized', 'inputs', 'targets_pretokenized', 'targets'])
  inp = data['inputs'] 
  decoded_inp = VOCABULARY.decode_tf(inp).numpy().decode('utf-8')
  tgt = data['targets'] 
  decoded_tgt = VOCABULARY.decode_tf(tgt).numpy().decode('utf-8')


  if does_task_reg_ds_have_pretokenized:
    inp_pre = data['inputs_pretokenized'].numpy().decode('utf-8')
    tgt_pre = data['targets_pretokenized'].numpy().decode('utf-8')


  print('\n')

  if does_task_reg_ds_have_pretokenized:
    print('======== input_pretokenized')
    print(inp_pre)

  print('======== VOCABULARY.decode_tf(tokenized_input)')
  print(decoded_inp)
  print(inp)

  if does_task_reg_ds_have_pretokenized:
    print('======== target_pretokenized')
    print(tgt_pre)

  print('======== VOCABULARY.decode_tf(tokenized_target)')
  print(decoded_tgt)
  print(tgt)
  print(f'{i+1} printed')

