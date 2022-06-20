import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
import seqio
from t5.data.tasks import TaskRegistry
from build_custom_datasets.utils import head_files
import numpy as np
task = sys.argv[1]

task_registry_ds = TaskRegistry.get_dataset(task, sequence_length={"inputs": 512, "targets": 512},
                                                    split='train', shuffle=False)



tmp = './t5/data/explore_downstream_tasks_data/task_lengths_stats'
os.system(f'mkdir -p {tmp}')
file_to_write_max_lengths_computed = open(osp.join(tmp, 
                                                    f'{task}_src_tgt_length_stats.txt'), mode='w')
max_inp_length = 0
max_tgt_length = 0
min_inp_length = 9999999
min_tgt_length = 9999999
inp_lengths = []
tgt_lengths = []
for i, data in enumerate(task_registry_ds.take(999999999)):
  inp = data['inputs'] 
  tgt = data['targets'] 
  # breakpoint()
  # if i == 0:
  #   print('-'*10) 
  #   print('printing first data')
  #   print(data.keys())
  #   print(f"inputs: {data['inputs']}")    
  #   print(f"targets: {data['targets']}")   
  #   print('-'*10) 
  # print(inp)
  max_inp_length = max(max_inp_length, len(inp))
  max_tgt_length = max(max_tgt_length, len(tgt))
  min_inp_length = min(min_inp_length, len(inp))
  min_tgt_length = min(min_tgt_length, len(tgt))

  inp_length = len(inp)
  tgt_length = len(tgt)
  if task == 'cnndm_from_pretraining_with_nonsense_paper':
    if inp_length > 512:
      inp_length = 512
    if tgt_length > 256:
      tgt_length = 256
  inp_lengths.append(inp_length)
  tgt_lengths.append(tgt_length)
  # print(i)
print('\n\n TRAIN ------')
print(len(inp_lengths))
file_to_write_max_lengths_computed.write(f'MAX INPUT LENGTH {max_inp_length}\n')
file_to_write_max_lengths_computed.write(f'MAX TGT LENGTH {max_tgt_length}\n')
file_to_write_max_lengths_computed.write(f'MIN INPUT LENGTH {min_inp_length}\n')
file_to_write_max_lengths_computed.write(f'MIN TGT LENGTH {min_tgt_length}\n')
print('-')
file_to_write_max_lengths_computed.write(f'MEAN INPUT LENGTH {np.mean(inp_lengths)}\n')
file_to_write_max_lengths_computed.write(f'MEAN TGT LENGTH {np.mean(tgt_lengths)}\n')
file_to_write_max_lengths_computed.close()
head_files([file_to_write_max_lengths_computed], 100)






# task_registry_ds = TaskRegistry.get_dataset(task, sequence_length={"inputs": 2048, "targets": 2048},
#                                                     split='validation', shuffle=False)



# tmp = './t5/data/explore_downstream_tasks_data/task_lengths_stats'
# os.system(f'mkdir -p {tmp}')
# file_to_write_max_lengths_computed = open(osp.join(tmp, 
#                                                     f'{task}_valid_src_tgt_length_stats.txt'), mode='w')
# max_inp_length = 0
# max_tgt_length = 0
# min_inp_length = 9999999
# min_tgt_length = 9999999
# inp_lengths = []
# tgt_lengths = []
# for i, data in enumerate(list(task_registry_ds.take(999999999))):
#   inp = data['inputs'] 
#   tgt = data['targets'] 
#   # if i == 0:
#   #   print('-'*10) 
#   #   print('printing first data')
#   #   print(data.keys())
#   #   print(f"inputs: {data['inputs']}")    
#   #   print(f"targets: {data['targets']}")   
#   #   print('-'*10) 
#   # print(inp)
#   max_inp_length = max(max_inp_length, len(inp))
#   max_tgt_length = max(max_tgt_length, len(tgt))
#   min_inp_length = min(min_inp_length, len(inp))
#   min_tgt_length = min(min_tgt_length, len(tgt))
#   inp_length = len(inp)
#   tgt_length = len(tgt)
#   if task == 'cnndm_from_pretraining_with_nonsense_paper':
#     if inp_length > 512:
#       inp_length = 512
#     if tgt_length > 256:
#       tgt_length = 256
#   inp_lengths.append(inp_length)
#   tgt_lengths.append(tgt_length)
# print('\n\n VALID ------')
# print(len(inp_lengths))

# file_to_write_max_lengths_computed.write(f'MAX INPUT LENGTH {max_inp_length}\n')
# file_to_write_max_lengths_computed.write(f'MAX TGT LENGTH {max_tgt_length}\n')
# file_to_write_max_lengths_computed.write(f'MEAN INPUT LENGTH {np.mean(inp_lengths)}\n')
# file_to_write_max_lengths_computed.write(f'MEAN TGT LENGTH {np.mean(tgt_lengths)}\n')
# file_to_write_max_lengths_computed.close()
# head_files([file_to_write_max_lengths_computed], 100)






# task_registry_ds = TaskRegistry.get_dataset(task, sequence_length={"inputs": 2048, "targets": 2048},
#                                                     split='test', shuffle=False)

# tmp = './t5/data/explore_downstream_tasks_data/task_lengths_stats'
# os.system(f'mkdir -p {tmp}')
# file_to_write_max_lengths_computed = open(osp.join(tmp, 
#                                                     f'{task}_test_src_tgt_length_stats.txt'), mode='w')
# max_inp_length = 0
# max_tgt_length = 0
# min_inp_length = 9999999
# min_tgt_length = 9999999
# inp_lengths = []
# tgt_lengths = []
# for i, data in enumerate(list(task_registry_ds.take(999999999))):
#   inp = data['inputs'] 
#   tgt = data['targets'] 
#   # if i == 0:
#   #   print('-'*10) 
#   #   print('printing first data')
#   #   print(data.keys())
#   #   print(f"inputs: {data['inputs']}")    
#   #   print(f"targets: {data['targets']}")   
#   #   print('-'*10) 
#   # print(inp)
#   max_inp_length = max(max_inp_length, len(inp))
#   max_tgt_length = max(max_tgt_length, len(tgt))
#   min_inp_length = min(min_inp_length, len(inp))
#   min_tgt_length = min(min_tgt_length, len(tgt))
#   inp_length = len(inp)
#   tgt_length = len(tgt)
#   if task == 'cnndm_from_pretraining_with_nonsense_paper':
#     if inp_length > 512:
#       inp_length = 512
#     if tgt_length > 256:
#       tgt_length = 256
#   inp_lengths.append(inp_length)
#   tgt_lengths.append(tgt_length)
# print('\n\n test ------')
# print(len(inp_lengths))

# file_to_write_max_lengths_computed.write(f'MAX INPUT LENGTH {max_inp_length}\n')
# file_to_write_max_lengths_computed.write(f'MAX TGT LENGTH {max_tgt_length}\n')
# file_to_write_max_lengths_computed.write(f'MEAN INPUT LENGTH {np.mean(inp_lengths)}\n')
# file_to_write_max_lengths_computed.write(f'MEAN TGT LENGTH {np.mean(tgt_lengths)}\n')
# file_to_write_max_lengths_computed.close()
# head_files([file_to_write_max_lengths_computed], 100)