r"""Code for loading the Python Meena synthesis dataset from bosma@.

To re-generate the TFDS dataset, use

OUTPUT_DIR=/cns/od-d/home/henrykm/rs=6.3/mathqa/tfds
blaze run -c opt \
learning/brain/research/program_synthesis/abcm/meena:download_and_prepare.par \
  -- \
  --data_dir="${OUTPUT_DIR}" \
  --datasets=mathqa_dataset \
  --module_import=google3.learning.brain.research.program_synthesis.abcm.meena.mathqa_dataset
  \
  --gfs_user=brain-synth \
  --flume_exec_mode=IN_PROCESS

Note: there seem to be two issues with this. First, it sometimes fails to find
the correct script if you use the .par file. Try not using the .par extension.
Second, sometimes it adds a fileFormat element to the dataset_info.json file.
Manually removing this can fix the dataset.
"""

import dataclasses
import functools
import json
import os
from typing import List, Tuple
import uuid

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

# from google3.pyglib import gfile
# from google3.pyglib import resources

DATASET_RESOURCE = '/Users/felix/Downloads/mathqa'

DATASET_PATH = '/Users/felix/tensorflow_datasets'


@dataclasses.dataclass
class DatasetEntry:
  prompt: str
  code: str
  dsl_code: str
  reasoning: str
  answer: float
  task_id: int


def format_entry(
    entry: DatasetEntry,
    include_code: bool = False,
    include_dsl_code: bool = False,
    prompt_template='Please, solve the mathematical problem: {prompt}'):
  """Converts a DatasetEntry to a string prompt for use as a model input.

  Args:
    entry: a DatasetEntry object.
    include_code: if True, will include the code in the prompt.
    include_dsl_code: if True, will include the dsl code in the prompt.
    prompt_template: the prompt template used to format prompts for this task.

  Returns:
    the string prompt generated from the entry.
  """

  prompt = prompt_template.format(prompt=entry.prompt)

  prompt += '\n\n[BEGIN]'

  if include_code:
    prompt += '\n\n' + entry.code + '\n\n[DONE]'

  if include_dsl_code:
    prompt += '\n\n' + entry.dsl_code + '\n\n[DONE]'

  return prompt


class MeenaPythonMathqaDataset:
  """A container class containing utilities for the MathQA dataset."""
  entries: List[DatasetEntry]

  def __init__(
      self,
      entries: List[DatasetEntry],
      prompt_template='Please, solve the mathematical problem: {prompt}',
  ):
    self.entries = entries
    self.prompt_template = prompt_template

  def __getitem__(self, idx):
    return self.entries[idx]

  def __len__(self):
    return len(self.entries)

  def __repr__(self):
    return f'MeenaPythonMathqaDataset(num_entries: {len(self)})'

  def create_prompt_prefix(
      self,
      num_examples: int = 4,
      start_idx=4,
      prefix='I know that you are very good at solving math problems.',
      dsl_code=False) -> str:
    """Creates the prompt prefix that contains prefix prompting examples.

    Args:
      num_examples: the number of examples to use for prompting.
      start_idx: the example to start with. We use this to skip the first
        example since it's kind of long.
      prefix: a prefix prefixed to the prompt.
      dsl_code: if true, then uses DSL code; otherwise uses Python.

    Returns:
      a string containing some number of example prompts.
    """

    prompt = prefix
    if dsl_code:
      format_entry_code = functools.partial(format_entry, include_dsl_code=True)
    else:
      format_entry_code = functools.partial(format_entry, include_code=True)
    for i in range(start_idx, start_idx + num_examples):
      prompt += format_entry_code(self.entries[i]) + '\n\n'

    return prompt

  def create_task(self,
                  idx: int,
                  few_shot: bool = False,
                  dsl_code: bool = False,
                  **kwargs) -> Tuple[str, DatasetEntry]:
    """Creates a prompt with a problem description.

    Args:
      idx: the index of the test problem to use.
      few_shot: if True, then add four initial examples.
      dsl_code: if True, then use DSL for inintial examples.
      **kwargs: kwargs passed to create_prompt_prefix.

    Returns:
      the prompt text containing problem description.
    """
    test_example = self.entries[idx]

    if few_shot:
      prompt = self.create_prompt_prefix(dsl_code=dsl_code, **kwargs)
    else:
      prompt = ''
    prompt += format_entry(
        test_example, include_code=False) + '\n\n'

    return prompt, test_example


def load_mathqa_dataset(
    prompt_template='Please, solve the mathematical problem: {prompt}',
    colab=False,
    train=True,
    eval=False,  # pylint: disable=redefined-builtin
    test=False,
    challenge=False):
  """Loads the Python dataset.

  Args:
    prompt_template: the template to use to generate prompts.
    colab: if True, will load the data in a way that supports Colab.
    train: if True, will load the training data.
    eval: if train is false and eval is True, will load the eval data.
    test: if train and eval are false and test is True, will load the test data.
    challenge: if train, eval and test are false and challenge is True,
      will load the training data, otherwise the challenge data.

  Returns:
    train, eval, test and challenge examples.
  """
  if (train + eval + test + challenge) != 1:
    raise ValueError('(train + eval + test + challenge) != 1')
  entries = []

  if train:
    path = os.path.join(DATASET_RESOURCE, 'python_train.jsonl')
  elif eval:
    path = os.path.join(DATASET_RESOURCE, 'python_dev.jsonl')
  elif test:
    path = os.path.join(DATASET_RESOURCE, 'python_test.jsonl')
  elif challenge:
    path = os.path.join(DATASET_RESOURCE, 'python_challenge_test.jsonl')
  else:
    raise ValueError('You need to set train, eval, test or challenge to true.')
  if colab:
    path = os.path.join('/google_src/head/depot/', path)
    with open(path, 'r') as file:
      f = file.readlines()
  else:
    f = open(path, mode = 'r')

  for line in f:
    entry = json.loads(line)
    entry = DatasetEntry(
        prompt=entry['text'],
        code=entry['code'],
        dsl_code=entry['dsl_code'],
        reasoning=entry['reasoning'],
        answer=entry['answer'],
        task_id=entry['task_id'],
    )

    entries.append(entry)

  f.close()
  ds = MeenaPythonMathqaDataset(entries, prompt_template=prompt_template)

  logging.info('WOOO! Loaded MathQA Python dataset with %d examples.',
               len(entries))
  return ds


class PythonMathqaDataset(tfds.core.GeneratorBasedBuilder):
  """Dataset of Math Q&A problems.

  See https://gitlab.cs.washington.edu/amini91/mathqa-categorization/
  for more details. Here we use the Python variant implemented previously
  in TRAX and the original DSL implementation.

  The train/eval/test/challenge split is suggested for finetuning. For zeroshot
  and fewshot eval use all splits.
  """

  VERSION = tfds.core.utils.Version('0.0.3')

  @property
  def description(self):
    return ('Dataset of math problems with solutions implemented using Python '
            'and a DSL.')

  @property
  def data_location(self):
    return '/google/src/head/depot/google3/learning/brain/research/program_synthesis/mathqa_dataset/'

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=self.description,
        features=tfds.features.FeaturesDict({
            'text': tfds.features.Text(),
            'code': tfds.features.Text(),
            'dsl_code': tfds.features.Text(),
            'reasoning': tfds.features.Text(),
            'answer': tf.float32,
            'task_id': tf.int32,
        }),
    )

  def _load_split(self, split_file_name):
    data_location = os.path.join(self.data_location, split_file_name)
    with open(data_location, mode = 'r') as f:
      data = list(map(json.loads, f))
    return data

  def _split_generators(self, dl_manager):
    train_data = self._load_split('python_train.jsonl')
    eval_data = self._load_split('python_dev.jsonl')
    test_data = self._load_split('python_test.jsonl')
    challenge_data = self._load_split('python_challenge_test.jsonl')

    # Shuffle data
    train_data.sort(key=lambda x: x['task_id'])

    return {
        'train': self._generate_examples(train_data),
        'eval': self._generate_examples(eval_data),
        'test': self._generate_examples(test_data),
        'challenge': self._generate_examples(challenge_data),
    }

  def _generate_examples(self, data):
    for item in data:
      yield str(uuid.uuid4()), item


def load_mathqa_dataset_tfds(
    prompt_template='Please, solve the mathematical problem: {prompt}'):
  """Loads the MathQA dataset.

  Args:
    prompt_template: the template to use to generate prompts.

  Returns:
    a train and test dataset partitioning the set of examples.
  """
  builder = PythonMathqaDataset(data_dir=DATASET_PATH)
  train_ds, eval_ds, test_ds, challenge_ds = builder.as_dataset(
      'train'), builder.as_dataset('eval'), builder.as_dataset(
          'test'), builder.as_dataset('challenge')

  def _load_from_tf_data(ds):
    entries = []
    for entry in ds.as_numpy_iterator():
      entries.append(
          DatasetEntry(
              prompt=entry['text'].decode(),
              code=entry['code'].decode(),
              dsl_code=entry['dsl_code'].decode(),
              reasoning=entry['reasoning'].decode(),
              answer=entry['answer'],
              task_id=entry['task_id'],
          ))

    ds = MeenaPythonMathqaDataset(entries, prompt_template=prompt_template)

    return ds

  train_ds, eval_ds, test_ds, challenge_ds = _load_from_tf_data(
      train_ds), _load_from_tf_data(eval_ds), _load_from_tf_data(
          test_ds), _load_from_tf_data(challenge_ds),

  logging.info('Loaded the tfds MathQA dataset with %d examples.',
               len(train_ds) + len(eval_ds) + len(test_ds) + len(challenge_ds))

  return train_ds, eval_ds, test_ds, challenge_ds





# @dataclasses.dataclass
# class DatasetEntry:
#   prompt: str
#   code: str
#   dsl_code: str
#   reasoning: str
#   answer: float
#   task_id: int


if __name__ == '__main__':
  eval_ds = load_mathqa_dataset(
    prompt_template='Please, solve the mathematical problem: {prompt}',
    colab=False,
    train=False,
    eval=True,  # pylint: disable=redefined-builtin
    test=False,
    challenge=False)

  ds_entry_attributes = ['prompt', 'code', 'dsl_code', 'reasoning', 'answer', 'task_id']
  for i in range(10):
    print("=================================")
    for a in ds_entry_attributes:
      print(f'---- {a}')
      print(eval(f'eval_ds[i].{a}'))