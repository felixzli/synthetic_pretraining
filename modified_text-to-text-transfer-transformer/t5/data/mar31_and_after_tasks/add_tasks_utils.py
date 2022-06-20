#####
#####
# you must fill this variable with your path to the data after running `bash run_scripts_to_reproduce_experiments/data_scripts/get_all_pretraining_and_finetuning_data_from_gcs.sh``
DATA_BASE_DIR = '/mnt/disks/persist/draft_synthetic_pretraining_code/data/data_from_gcs/'
#
#
#
#


import functools
import seqio
import tensorflow as tf


DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100
SPM = {
    "t5": seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS),
    "ben_src": seqio.SentencePieceVocabulary('/mnt/disks/persist/quantum_datagen_2/sentencepiece_models/src_spm.model', 0),
    "ben_tgt": seqio.SentencePieceVocabulary('/mnt/disks/persist/quantum_datagen_2/sentencepiece_models/tgt_spm.model', 0),
    "isarstep": seqio.SentencePieceVocabulary('/mnt/disks/persist/felix-text-to-text-transfer-transformer/t5/data/mar31_and_after_tasks/sentencepiece_models/isarstep.model', 0)
}


from t5.data import get_default_vocabulary

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=get_default_vocabulary(), add_eos=True)
}


sentencepiece_model_file = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"
vocab = seqio.SentencePieceVocabulary(sentencepiece_model_file)


def tokenizer(spm_key):
  """Return the appropriate tokenizer given the vocabulary."""
  return functools.partial(
      seqio.preprocessors.tokenize_and_append_eos,
      output_features=t5_output_features(spm_key),
      copy_pretokenized=False)


def t5_output_features(spm_key, input_keys=("inputs", "targets")):
  """Return the output features for the given vocabulary."""
  return {x: seqio.Feature(SPM[spm_key], add_eos=True) for x in input_keys}


def ben_output_features():
    return {"inputs": seqio.Feature(SPM['ben_src'], add_eos=True), "targets": seqio.Feature(SPM['ben_tgt'], add_eos=True)}


def translation_processors():
  """Prepare texts for translation tasks."""
  split_map_fn = lambda x: tf.strings.split(x, sep="\t", maxsplit=-1)
  rekey_map_fn = lambda x: {"inputs": x[0], "targets": x[1]}

  def split(dataset):
    dataset = dataset.map(split_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # Remove items with more than one tab per line.
    dataset = dataset.filter(lambda x: tf.shape(x)[0] == 2)
    dataset = dataset.map(rekey_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

  processors = [
      split,
      tokenizer("t5"),
  ]
  return processors


  #################


def lime_processors():
  """Prepare texts for translation tasks."""
  split_map_fn = lambda x: tf.strings.split(x, sep="\t", maxsplit=-1)
  rekey_map_fn = lambda x: {"inputs": x[0], "targets": x[1]}
    
  def split_and_rekey(dataset):
    dataset = dataset.map(split_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # Remove items with more than one tab per line.
    dataset = dataset.filter(lambda x: tf.shape(x)[0] == 2)
    # breakpoint()
    # dataset = dataset.map(help, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(rekey_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # breakpoint()
    return dataset

  processors = [
      split_and_rekey,
      lime_process_ds
  ]

  return processors


def artificial_language_processors():
  """Prepare texts for translation tasks."""
  split_map_fn = lambda x: x
  rekey_map_fn = lambda x: {"inputs": None, "targets": x}
    
  def split_and_rekey(dataset):
    dataset = dataset.map(split_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # Remove items with more than one tab per line.
    # breakpoint()
    # dataset = dataset.map(help, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(rekey_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # breakpoint()
    return dataset


  processors = [
      split_and_rekey,
      functools.partial(lime_process_ds, is_add_eos=False)
  ]

  return processors



def _lime_data_process_helper_func(x):
  x = tf.strings.substr(x, 1, tf.strings.length(x) - 2)
  x = tf.strings.split(x, ', ')  
  x = tf.strings.to_number(x, out_type=tf.dtypes.int32)
  return x


def _lime_process_one_features(features, is_add_eos):
  ret = {}
  for k, v in features.items():
    if v is None:
      continue
    if k in ['inputs', 'targets']:
      v = _lime_data_process_helper_func(v)
      if is_add_eos:
        # Expand dims here so that the below code can work with 1-d tensors.
        v = tf.expand_dims(v, 0)
        # Make sure we keep tensor as ragged to allow for uneven concat.
        if isinstance(v, tf.Tensor):
          v = tf.RaggedTensor.from_tensor(v)

        # Append eos to the last item of every sequence.
        eos_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
        eos_id = tf.broadcast_to(vocab.eos_id, eos_shape)
        last_in_sequence = tf.concat([v[..., -1:, :], eos_id], axis=-1)
        # Concat back the newly modified final sequence item.
        v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)
        # Un-expand outer dimension.
        v = v[0]
    ret[k] = v
  return ret


def lime_process_ds(ds, is_add_eos=True):
  return ds.map(lambda x: _lime_process_one_features(x, is_add_eos), num_parallel_calls=tf.data.AUTOTUNE)