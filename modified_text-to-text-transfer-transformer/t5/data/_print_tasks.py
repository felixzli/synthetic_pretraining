import tasks
from tasks import seqio


# basic_unary_tasks_dataset = seqio.get_mixture_or_task("copy_1_11_t5_token_basic_unary_tasks").get_dataset(
# # dataset = seqio.get_mixture_or_task("induct_100k_1_23_lime").get_dataset(
#   sequence_length={"inputs": 256, "targets": 128},
#   split="train",
#   shuffle=True,
#   num_epochs=1,
#   shard_info=seqio.ShardInfo(index=0, num_shards=10),
#   use_cached=False,
#   seed=42
# )

# # Print the first 5 examples.
# for _, ex in zip(range(2), basic_unary_tasks_dataset.as_numpy_iterator()):
#   print(ex)
# print('=================')
# # dataset = seqio.get_mixture_or_task("1_11_t5_token_basic_unary_tasks").get_dataset(
# dataset = seqio.get_mixture_or_task("induct_100k_1_23_lime").get_dataset(
#   sequence_length={"inputs": 256, "targets": 128},
#   split="train",
#   shuffle=True,
#   num_epochs=1,
#   shard_info=seqio.ShardInfo(index=0, num_shards=10),
#   use_cached=False,
#   seed=42
# )


# # Print the first 5 examples.
# for _, ex in zip(range(2), dataset.as_numpy_iterator()):
#   print(ex)

# dataset = seqio.get_mixture_or_task("abduct_100k_1_23_lime").get_dataset(
#   sequence_length={"inputs": 256, "targets": 128},
#   split="train",
#   shuffle=True,
#   num_epochs=1,
#   shard_info=seqio.ShardInfo(index=0, num_shards=10),
#   use_cached=False,
#   seed=42
# )


# # Print the first 5 examples.
# for _, ex in zip(range(2), dataset.as_numpy_iterator()):
#   print(ex)


# dataset = seqio.get_mixture_or_task("deduct_100k_1_23_lime").get_dataset(
#   sequence_length={"inputs": 256, "targets": 128},
#   split="train",
#   shuffle=True,
#   num_epochs=1,
#   shard_info=seqio.ShardInfo(index=0, num_shards=10),
#   use_cached=False,
#   seed=42
# )


# # Print the first 5 examples.
# for _, ex in zip(range(2), dataset.as_numpy_iterator()):
#   print(ex)



dataset = seqio.get_mixture_or_task("100k_1_23_lime").get_dataset(
  sequence_length={"inputs": 256, "targets": 128},
  split="train",
  shuffle=True,
  num_epochs=1,
  shard_info=seqio.ShardInfo(index=0, num_shards=10),
  use_cached=False,
  seed=42
)


# Print the first 5 examples.
for _, ex in zip(range(2), dataset.as_numpy_iterator()):
  print(ex)

  # 1M_1_23_lime


dataset = seqio.get_mixture_or_task("1M_1_23_lime").get_dataset(
  sequence_length={"inputs": 256, "targets": 256},
  split="train",
  shuffle=True,
  num_epochs=1,
  shard_info=seqio.ShardInfo(index=0, num_shards=10),
  use_cached=False,
  seed=42
)


# Print the first 5 examples.
for _, ex in zip(range(2), dataset.as_numpy_iterator()):
  print(ex)