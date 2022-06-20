import numpy as np
import tasks
from tasks import seqio

import json
import os.path as osp
import os

def count_lists_with_last_element_eq_1(lsts):
    c = 0
    for lst in lsts:
        if lst[-1] == 1:
            c += 1
    return c



EXTRA_TOKEN_IDS = list(range(32099,31999,-1))

def is_lst_wiki_data(lst):
  return lst[-1] == 1 and all([x in lst for x in EXTRA_TOKEN_IDS[:6]])

def count_wiki_data(lsts):
    c = 0
    for lst in lsts:
        if is_lst_wiki_data(lst):
            c += 1
    return c


def print_wiki_examples_per_batch(tasks_to_print, is_pack, n, file_to_write, length=512, bs=128):

  for task in tasks_to_print:
    print("\n*(@!*&#!(@*&#(*!@&#(*!@&(*#&*!@(*#&!@ ")
    print(task)
    print(f'PACK:{is_pack}')
    print("*(@!*&#!(@*&#(*!@&#(*!@&(*#&*!@(*#&!@ ")
    # dataset = seqio.get_mixture_or_task(t).get_dataset
    
    dataset = seqio.get_dataset(
      mixture_or_task_name=task,
      task_feature_lengths={"inputs": length, "targets": length},
      dataset_split="train",
      shuffle=True,
      num_epochs=1,
      shard_info=seqio.ShardInfo(index=0, num_shards=10),
      use_cached=False,
      seed=42,
      feature_converter=seqio.EncDecFeatureConverter(
          pack=is_pack, use_custom_packing_ops=False),
    )

    path = osp.join(osp.dirname(file_to_write), f'{task}_{osp.basename(file_to_write)}')
    f = open(path, mode='w')
    dataset = dataset.batch(bs, drop_remainder=True)

    wiki_data_counts = []
    last_one_counts = []
    wiki_target_tokens_counts=[]
    for _, ex in zip(range(n), dataset.as_numpy_iterator()):
      for k,v in ex.items():
        ex[k] = v.tolist()

      wttc = 0
      wdc = 0
      for input, dlw in zip(ex['encoder_input_tokens'], ex['decoder_loss_weights']):
        if is_lst_wiki_data(input):
          wdc += 1
          wttc += sum(dlw)

      one_batch_wiki_data_count = count_wiki_data(ex['encoder_input_tokens'])
      assert one_batch_wiki_data_count == wdc
      one_batch_last_one_data_count = count_lists_with_last_element_eq_1(ex['encoder_input_tokens'])

      wiki_target_tokens_counts.append(wttc)
      wiki_data_counts.append(one_batch_wiki_data_count)
      last_one_counts.append(one_batch_last_one_data_count)
    f.write('wiki_data_counts\n')
    f.write(str(wiki_data_counts))
    f.write('\n')
    f.write('last_one_counts\n')
    f.write(str(last_one_counts))
    f.write('\n')
    f.write(f"AVERAGE WIKI DATA EXAMPLE PER BATCH (over {n} batches of batch size {bs})")
    f.write('\n')
    f.write(str(np.array(wiki_data_counts).mean()))
    f.write('\n')
    f.write(f"AVERAGE LAST ONE DATA EXAMPLE PER BATCH (over {n} batches of batch size {bs})")
    f.write('\n')
    f.write(str(np.array(last_one_counts).mean()))
    f.write('\n')
    f.write(f"AVERAGE NUM WIKI DECODER TOKENS (over {n} batches of batch size {bs})")
    f.write('\n')
    f.write(str(np.array(wiki_target_tokens_counts).mean()))
    f.write('\n')
    f.close()
    os.system(f'tail -n 6 {path}')

# mixtures = ['p5_5_wiki40b_and_1_23', 
#             'p25_75_wiki40b_and_1_23',
#             'p125_875_wiki40b_and_1_23']


mixtures = ['p1_0_wiki40b_and_1_23']
# print_wiki_examples_per_batch(mixtures, False, 1000, './tmp/pack_false_count_num_wiki_examples_per_batch.txt')
# print_wiki_examples_per_batch(mixtures, True, 1000, './tmp/pack_true_count_num_wiki_examples_per_batch.txt')
print_wiki_examples_per_batch(mixtures, True, 1000, './tmp/pack_true_count_num_wiki_examples_per_batch.txt')

