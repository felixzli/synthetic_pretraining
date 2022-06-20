import tasks
from tasks import seqio


def print_tasks(tasks_to_print):
  for t in tasks_to_print:
    print("\n\n\n\n\n\n*(@!*&#!(@*&#(*!@&#(*!@&(*#&*!@(*#&!@ ")
    print(t)
    print("*(@!*&#!(@*&#(*!@&#(*!@&(*#&*!@(*#&!@ ")
    dataset = seqio.get_mixture_or_task(t).get_dataset(
      sequence_length={"inputs": 256, "targets": 256},
      split="train",
      shuffle=True,
      num_epochs=1,
      shard_info=seqio.ShardInfo(index=0, num_shards=10),
      use_cached=False,
      seed=42
    )

    for _, ex in zip(range(3), dataset.as_numpy_iterator()):
      print(ex)


unary_tasks = ['copy', 'reverse', 'set', 
'first_char', 'last_char', 
'length', 'duplicate', 'deduplicate', 'longest_word']
# tasks = ['first_char']

unary_tasks = [task+ "_1M_1_23_unary" for task in unary_tasks]

print_tasks(unary_tasks)
# breakpoint()
mixtures = ["1M_1_23_lime", "1M_1_23_unary", '1M_1_23_lime_AND_unary', 'wiki40b',
            'p5_5_wiki40b_and_1_23', 
            'p25_75_wiki40b_and_1_23',
            'p125_875_wiki40b_and_1_23']


print_tasks(mixtures)