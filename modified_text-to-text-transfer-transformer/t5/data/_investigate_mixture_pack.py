import tasks
from tasks import seqio

import json



def print_tasks(tasks_to_print, is_pack, n, file_to_write, length=512, bs=32):
  f = open(file_to_write, mode='w')

  for t in tasks_to_print:
    print("\n\n\n\n\n\n*(@!*&#!(@*&#(*!@&#(*!@&(*#&*!@(*#&!@ ")
    print(t)
    print("*(@!*&#!(@*&#(*!@&#(*!@&(*#&*!@(*#&!@ ")
    # dataset = seqio.get_mixture_or_task(t).get_dataset
    
    dataset = seqio.get_dataset(
      mixture_or_task_name=t,
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
    
    dataset = dataset.batch(bs, drop_remainder=True)

    for _, ex in zip(range(n), dataset.as_numpy_iterator()):
      # f.write(str(ex))
      # f.write('\n')
      for k,v in ex.items():
        ex[k] = v.tolist()
      print(ex)
      print(ex, file=f)
  f.close()

mixtures = ['p5_5_wiki40b_and_1_23']

print_tasks(mixtures, True, 2, './tmp/pack_true_bs32_p5_5_wiki40b_and_1_23.txt')

print('9999999999999999999999999999999999999')
print('9999999999999999999999999999999999999')

print_tasks(mixtures, False, 2, './tmp/pack_false_bs32_p5_5_wiki40b_and_1_23.txt')



print_tasks(mixtures, False, 2, './tmp/length2048_pack_false_bs32_p5_5_wiki40b_and_1_23.txt', 2048)
print_tasks(mixtures, True, 2, './tmp/length2048_pack_true_bs32_p5_5_wiki40b_and_1_23.txt', 2048)


print_tasks(mixtures, False, 2, './tmp/length8192_pack_false_bs8_p5_5_wiki40b_and_1_23.txt', 8192, 8)
print_tasks(mixtures, True, 2, './tmp/length8192_pack_true_bs8_p5_5_wiki40b_and_1_23.txt', 8192, 8)


print_tasks(mixtures, False, 2, './tmp/length2048_pack_false_bs100_p5_5_wiki40b_and_1_23.txt', 2048, 100)
print_tasks(mixtures, True, 2, './tmp/length2048_pack_true_bs100_p5_5_wiki40b_and_1_23.txt', 2048, 100)




print_tasks(mixtures, False, 100, './tmp/length512_pack_false_bs128_p5_5_wiki40b_and_1_23.txt', 512, 128)
print_tasks(mixtures, True, 100, './tmp/length512_pack_true_bs128_p5_5_wiki40b_and_1_23.txt', 512, 128)
