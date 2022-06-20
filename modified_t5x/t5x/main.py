# Copyright 2022 The T5X Authors.
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

r"""The main entrance for running any of the T5X supported binaries.

Currently this includes train/infer/eval/precompile.

Example Local (CPU) Pretrain Gin usage

python -m t5x.main \
  --gin_file=t5x/examples/t5/t5_1_1/tiny.gin \
  --gin_file=t5x/configs/runs/pretrain.gin \
  --gin.MODEL_DIR=\"/tmp/t5x_pretrain\" \
  --gin.TRAIN_STEPS=10 \
  --gin.MIXTURE_OR_TASK_NAME=\"c4_v220_span_corruption\" \
  --gin.MIXTURE_OR_TASK_MODULE=\"t5.data.mixtures\" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 128, 'targets': 30}" \
  --gin.DROPOUT_RATE=0.1 \
  --run_mode=train \
  --logtostderr
"""
import enum
import os
from typing import Sequence

from absl import app
from absl import flags

import gin
import jax
import seqio

from t5x import eval as eval_lib
from t5x import gin_utils
from t5x import infer as infer_lib
from t5x import precompile as precompile_lib
from t5x import train as train_lib
from t5x import utils



@enum.unique
class RunMode(enum.Enum):
  """All the running mode possible in T5X."""
  TRAIN = 'train'
  EVAL = 'eval'
  INFER = 'infer'
  PRECOMPILE = 'precompile'


_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file',
    default=None,
    help='Path to gin configuration file. Multiple paths may be passed and '
    'will be imported in the given order, with later configurations  '
    'overriding earlier ones.')

_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', default=[], help='Individual gin bindings.')

_GIN_SEARCH_PATHS = flags.DEFINE_list(
    'gin_search_paths',
    default=['.'],
    help='Comma-separated list of gin config path prefixes to be prepended '
    'to suffixes given via `--gin_file`. If a file appears in. Only the '
    'first prefix that produces a valid path for each suffix will be '
    'used.')

_RUN_MODE = flags.DEFINE_enum_class(
    'run_mode',
    default=None,
    enum_class=RunMode,
    help='The mode to run T5X under')

_TFDS_DATA_DIR = flags.DEFINE_string(
    'tfds_data_dir', None,
    'If set, this directory will be used to store datasets prepared by '
    'TensorFlow Datasets that are not available in the public TFDS GCS '
    'bucket. Note that this flag overrides the `tfds_data_dir` attribute of '
    'all `Task`s.')

_DRY_RUN = flags.DEFINE_bool(
    'dry_run', False,
    'If set, does not start the function but stil loads and logs the config.')


FLAGS = flags.FLAGS

# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

train = train_lib.train
evaluate = eval_lib.evaluate
infer = infer_lib.infer
precompile = precompile_lib.precompile

_FUNC_MAP = {
    RunMode.TRAIN: train,
    RunMode.EVAL: evaluate,
    RunMode.INFER: infer,
    RunMode.PRECOMPILE: precompile,
}


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  if _TFDS_DATA_DIR.value is not None:
    seqio.set_tfds_data_dir_override(_TFDS_DATA_DIR.value)

  # Register function explicitly under __main__ module, to maintain backward
  # compatability of existing '__main__' module references.
  gin.register(_FUNC_MAP[_RUN_MODE.value], '__main__')

  gin_utils.parse_gin_flags(
      # User-provided gin paths take precedence if relative paths conflict.
      _GIN_SEARCH_PATHS.value + _DEFAULT_GIN_SEARCH_PATHS,
      _GIN_FILE.value,
      _GIN_BINDINGS.value)

  if _DRY_RUN.value:
    return

  run_with_gin = gin.get_configurable(_FUNC_MAP[_RUN_MODE.value])
  run_with_gin()


def _flags_parser(args: Sequence[str]) -> Sequence[str]:
  """Flag parser.

  See absl.app.parse_flags_with_usage and absl.app.main(..., flags_parser).

  Args:
    args: All command line arguments.

  Returns:
    [str], a non-empty list of remaining command line arguments after parsing
    flags, including program name.
  """
  return app.parse_flags_with_usage(list(gin_utils.rewrite_gin_args(args)))


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  app.run(main, flags_parser=_flags_parser)
