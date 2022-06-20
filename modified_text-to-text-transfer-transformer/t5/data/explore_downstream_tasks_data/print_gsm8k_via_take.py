import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
import seqio
from t5.data.tasks import TaskRegistry
# from t5.data.print_downstream_tasks_data import print_any_task_via_take

os.system('python t5/data/print_downstream_tasks_data/print_any_task_via_take.py gsm8k 0')

# task = 'gsm8k'
# task_registry_ds = {}
# for split in ['validation']:
#   task_registry_ds[split] = TaskRegistry.get_dataset(task, sequence_length=None,
#                                                     split=split, shuffle=False)


# DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
# DEFAULT_EXTRA_IDS = 100

# SPM = {
#     "t5": seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)
# }

# VOCABULARY = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


# for data in list(task_registry_ds['validation'].take(10)):
#   # data.keys() ---> dict_keys(['inputs_pretokenized', 'inputs', 'targets_pretokenized', 'targets'])
#   print(data.keys())
#   # inp_pre = data['inputs_pretokenized'].numpy().decode('utf-8')
#   inp = data['inputs'] 

#   decoded_inp = VOCABULARY.decode_tf(inp).numpy().decode('utf-8')
#   # tgt_pre = data['targets_pretokenized'].numpy().decode('utf-8')
#   tgt = data['targets'] 
#   decoded_tgt = VOCABULARY.decode_tf(tgt).numpy().decode('utf-8')

#   print('\n')
#   # print('======== input_pretokenized')
#   # print(inp_pre)
#   print('======== VOCABULARY.decode_tf(tokenized_input)')
#   print(decoded_inp)

#   # print('======== target_pretokenized')
#   # print(tgt_pre)
#   print('======== VOCABULARY.decode_tf(tokenized_target)')
#   print(decoded_tgt)

