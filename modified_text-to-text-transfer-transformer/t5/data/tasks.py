# Copyright 2021 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add Tasks to registry."""
# TODO(adarob): Switch to seqio.Task.

import functools
from build.lib.t5.evaluation.qa_utils import qa_metrics

import seqio
import t5.data
from t5.data import postprocessors
from t5.data import preprocessors
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from t5.evaluation import metrics, qa_utils
import tensorflow_datasets as tfds
import tensorflow as tf

TaskRegistry = seqio.TaskRegistry



DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}

# ==================================== C4 ======================================
# Final pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "c4_v220_span_corruption",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# Baseline pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "c4_v220_iid_denoising",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.iid_denoising,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# Prefix language modeling pretraining task used in Raffel et al., 2019.
TaskRegistry.add(
    "c4_v220_prefix_lm",
    # source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    source=seqio.TfdsDataSource(tfds_name="c4/en:3.0.1"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.prefix_lm,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# Configurable tasks used for comparisons in Raffel et al., 2019.
_c4_config_suffixes = ["", ".noclean", ".realnewslike", ".webtextlike"]
for config_suffix in _c4_config_suffixes:
  TaskRegistry.add(
      "c4{name}_v020_unsupervised".format(name=config_suffix.replace(".", "_")),
      source=seqio.TfdsDataSource(tfds_name="c4/en{config}:2.2.0".format(
          config=config_suffix)),
      preprocessors=[
          functools.partial(
              preprocessors.rekey, key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          preprocessors.unsupervised,
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])


# ================================ Wikipedia ===================================
TaskRegistry.add(
    "wikipedia_20190301.en_v003_unsupervised",
    source=seqio.TfdsDataSource(tfds_name="wikipedia/20190301.en:1.0.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.unsupervised,
        # preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


TaskRegistry.add(
    "wikipedia_20190301.en_v003_span_corruption",
    source=seqio.TfdsDataSource(tfds_name="wikipedia/20190301.en:1.0.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        # preprocessors.unsupervised,
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


TaskRegistry.add(
    "wiki40b",
    source=seqio.TfdsDataSource(tfds_name="wiki40b/en:1.3.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        # preprocessors.unsupervised,
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


# =================================== GLUE =====================================
for b in tfds.text.glue.Glue.builder_configs.values():
  TaskRegistry.add(
      "glue_%s_v002" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="glue/%s:1.0.0" % b.name,
          splits=["test"] if b.name == "ax" else None),
      preprocessors=[
          get_glue_text_preprocessor(b),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=get_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b))

# =============================== CNN DailyMail ================================
TaskRegistry.add(
    "cnn_dailymail_v002",
    # source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.1.0"),
    source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.2.0"),

    preprocessors=[
        functools.partial(
            preprocessors.summarize,
            article_key="article",
            summary_key="highlights"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ==================================== WMT =====================================
# Format: year, tfds builder config, tfds version
b_configs = [
    ("14", tfds.translate.wmt14.Wmt14Translate.builder_configs["de-en"], "1.0.0"
    ),
    ("14", tfds.translate.wmt14.Wmt14Translate.builder_configs["fr-en"], "1.0.0"
    ),
    ("16", tfds.translate.wmt16.Wmt16Translate.builder_configs["ro-en"], "1.0.0"
    ),
    ("15", tfds.translate.wmt15.Wmt15Translate.builder_configs["fr-en"], "1.0.0"
    ),
    ("19", tfds.translate.wmt19.Wmt19Translate.builder_configs["de-en"], "1.0.0"
    ),
    ("17", tfds.translate.wmt17.Wmt17Translate.builder_configs["de-en"], "1.0.0")
]

for prefix, b, tfds_version in b_configs:
  TaskRegistry.add(
      "wmt%s_%s%s_v003" % (prefix, b.language_pair[1], b.language_pair[0]),
      source=seqio.TfdsDataSource(tfds_name="wmt%s_translate/%s:%s" %
                                  (prefix, b.name, tfds_version)),
      preprocessors=[
          functools.partial(
              preprocessors.translate,
              source_language=b.language_pair[1],
              target_language=b.language_pair[0],
          ),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[metrics.bleu],
      output_features=DEFAULT_OUTPUT_FEATURES)

# Special case for t2t ende.
b = tfds.translate.wmt_t2t.WmtT2tTranslate.builder_configs["de-en"]
TaskRegistry.add(
    "wmt_t2t_ende_v003",
    source=seqio.TfdsDataSource(tfds_name="wmt_t2t_translate/de-en:1.0.0"),
    preprocessors=[
        functools.partial(
            preprocessors.translate,
            source_language=b.language_pair[1],
            target_language=b.language_pair[0]),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.bleu],
    output_features=DEFAULT_OUTPUT_FEATURES)

## please :)




##
# ================================= SuperGlue ==================================
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
  # We use a simplified version of WSC, defined below
  if "wsc" in b.name:
    continue
  if b.name == "axb":
    glue_preprocessors = [
        functools.partial(
            preprocessors.rekey,
            key_map={
                "premise": "sentence1",
                "hypothesis": "sentence2",
                "label": "label",
                "idx": "idx",
            }),
        get_glue_text_preprocessor(b),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ]
  else:
    glue_preprocessors = [
        get_glue_text_preprocessor(b),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ]
  TaskRegistry.add(
      "super_glue_%s_v102" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="super_glue/%s:1.0.2" % b.name,
          splits=["test"] if b.name in ["axb", "axg"] else None),
      preprocessors=glue_preprocessors,
      metric_fns=get_super_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b))

  # Create SuperGLUE tasks with 1 sentinel token added.
  seqio.experimental.add_task_with_sentinels("super_glue_%s_v102" % b.name,
                                             num_sentinels=1)

# ======================== Definite Pronoun Resolution =========================
TaskRegistry.add(
    "dpr_v001_simple",
    source=seqio.TfdsDataSource(tfds_name="definite_pronoun_resolution:1.1.0"),
    preprocessors=[
        preprocessors.definite_pronoun_resolution_simple,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("dpr_v001_simple", num_sentinels=1)

# =================================== WSC ======================================
TaskRegistry.add(
    "super_glue_wsc_v102_simple_train",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
    preprocessors=[
        functools.partial(preprocessors.wsc_simple, correct_referent_only=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_train",
                                           num_sentinels=1)

TaskRegistry.add(
    "super_glue_wsc_v102_simple_eval",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation", "test"]),
    preprocessors=[
        functools.partial(
            preprocessors.wsc_simple, correct_referent_only=False),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)
# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_eval",
                                           num_sentinels=1)

# =================================== WNLI =====================================
TaskRegistry.add(
    "glue_wnli_v002_simple_eval",
    source=seqio.TfdsDataSource(
        tfds_name="glue/wnli:1.0.0", splits=["validation", "test"]),
    preprocessors=[
        preprocessors.wnli_simple,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# =================================== Squad ====================================
# Maximized evaluation metrics over all answers.
TaskRegistry.add(
    "squad_v010_allanswers",
    source=seqio.TfdsDataSource(tfds_name="squad/v1.1:3.0.0"),
    preprocessors=[
        preprocessors.squad,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)


# Maximized evaluation metrics over all answers.
TaskRegistry.add(
    "squad_v010_context_free",
    source=seqio.TfdsDataSource(tfds_name="squad/v1.1:3.0.0"),
    preprocessors=[
        functools.partial(preprocessors.squad, include_context=False),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Squad span prediction task instead of text.
TaskRegistry.add(
    "squad_v010_allanswers_span",
    source=seqio.TfdsDataSource(tfds_name="squad/v1.1:3.0.0"),
    preprocessors=[
        preprocessors.squad_span_space_tokenized,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.span_qa,
    metric_fns=[metrics.span_squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Deprecated: Use `squad_v010_allanswers` instead.
TaskRegistry.add(
    "squad_v010",
    source=seqio.TfdsDataSource(tfds_name="squad/v1.1:3.0.0"),
    preprocessors=[
        preprocessors.squad,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ================================= TriviaQA ===================================
TaskRegistry.add(
    "trivia_qa_v010",
    source=seqio.TfdsDataSource(tfds_name="trivia_qa/rc:1.1.0"),
    preprocessors=[
        preprocessors.trivia_qa,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.trivia_qa_truncate_inputs,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES)


# =============== PrefixLM objectives (not used in the T5 paper) ===============


# Vocabulary (shared by encoder and decoder)
sentencepiece_model_file = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"

vocab = seqio.SentencePieceVocabulary(sentencepiece_model_file)

seqio.TaskRegistry.add(
    "c4_prefix_lm_objective_encoder_decoder_architecture",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.targets_for_prefix_lm_objective,
        preprocessors.pack_prefix_lm_encoder_decoder,
    ],
    output_features={
        "encoder_input_tokens": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_target_tokens": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_input_tokens": seqio.Feature(vocabulary=vocab, add_eos=False),
        "encoder_segment_ids": seqio.Feature(vocabulary=vocab, add_eos=False),
        "encoder_positions": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_segment_ids": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_positions": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_loss_weights": seqio.Feature(vocabulary=vocab, add_eos=False),
        # All but the last stage of the preprocessing uses "targets" as the key,
        # so this output feature is necessary. It is not marked required because
        # the final preprocessor drops it.
        "targets": seqio.Feature(vocabulary=vocab, required=False),
    },
    metric_fns=[])


seqio.TaskRegistry.add(
    "c4_prefix_lm_objective_decoder_architecture",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.targets_for_prefix_lm_objective,
        preprocessors.pack_prefix_lm_decoder_only,
    ],
    output_features={
        "decoder_target_tokens": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_input_tokens": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_loss_weights": seqio.Feature(vocabulary=vocab, add_eos=False),
        "decoder_causal_attention": seqio.Feature(
            vocabulary=vocab, add_eos=False),
        # All but the last stage of the preprocessing uses "targets" as the key,
        # so this output feature is necessary. It is not marked required because
        # the final preprocessor drops it.
        "targets": seqio.Feature(vocabulary=vocab, required=False),
    },
    metric_fns=[])


# =============== pretraining with nonsense tasks https://arxiv.org/pdf/2109.04953.pdf ==========================


DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100

SPM = {
    "t5": seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS),
    "ben_src": seqio.SentencePieceVocabulary('/mnt/disks/persist/quantum_datagen_2/sentencepiece_models/src_spm.model', 0),
    "ben_tgt": seqio.SentencePieceVocabulary('/mnt/disks/persist/quantum_datagen_2/sentencepiece_models/tgt_spm.model', 0)
}


def output_features(spm_key, input_keys=("inputs", "targets")):
  """Return the output features for the given vocabulary."""
  return {x: seqio.Feature(SPM[spm_key], add_eos=True) for x in input_keys}


def ben_output_features():
    return {"inputs": seqio.Feature(SPM['ben_src'], add_eos=True), "targets": seqio.Feature(SPM['ben_tgt'], add_eos=True)}


def tokenizer(spm_key):
  """Return the appropriate tokenizer given the vocabulary."""
  return functools.partial(
      seqio.preprocessors.tokenize_and_append_eos,
      output_features=output_features(spm_key),
      copy_pretokenized=False)


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



#####################################################################

import sys
import os
sys.path.append(os.path.dirname(__file__))
from mar31_and_after_tasks.add_downstream_tasks import add_custom_downstream_task, custom_downstream_tasks
from mar31_and_after_tasks.add_pretrain_tasks import add_nonsense_summary, add_nesting_language, add_lime, add_simpler_tasks

for t in custom_downstream_tasks:
    # print(t)
    add_custom_downstream_task(seqio.TaskRegistry, t)

add_nesting_language(seqio.TaskRegistry)
add_lime(seqio.TaskRegistry, seqio.MixtureRegistry)
add_nonsense_summary(seqio.TaskRegistry)
add_simpler_tasks(seqio.TaskRegistry)

