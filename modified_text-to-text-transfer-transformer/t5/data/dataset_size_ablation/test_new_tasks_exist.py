import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
import seqio
from t5.data.tasks import TaskRegistry
from build_custom_datasets.utils import head_files
import numpy as np
# task = sys.argv[1]
from os import system as bash

tasks = ['squad_1K', 'squad_10K', 'cnndm_1K', 'cnndm_100K', 'retrosynthesis_1K', 'retrosynthesis_10K']

for mixture_or_task_name in tasks:
  task_registry_ds = TaskRegistry.get_dataset(mixture_or_task_name, sequence_length={"inputs": 10, "targets": 10},
                                                    split='train', shuffle=False)
  i = 0
  for _ in task_registry_ds.take(999999):
    i += 1

  print(f'{mixture_or_task_name} exists and has length {i}')


# squad_1K exists and has length 1000                                                                                                                                                                       
# squad_10K exists and has length 10000                                                                                                                                                                     
# cnndm_1K exists and has length 1000                                                                                                                                                                       
# cnndm_100K exists and has length 99998                                                                                                                                                                    
# retrosynthesis_1K exists and has length 1000                                                                                                                                                              
# retrosynthesis_10K exists and has length 10000   