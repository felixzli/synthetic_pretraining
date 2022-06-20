import seqio
task = 'cnn_dailymail_v002'
# task = 'hol_light_mlm'
# ds_provider = seqio.get_mixture_or_task(task)

from tasks import TaskRegistry
# TaskRegistry = seqio.TaskRegistry

task_registry_ds = {}
for split in ['validation']:
  task_registry_ds[split] = TaskRegistry.get_dataset(task, sequence_length=None,
                                                    split=split, shuffle=False)



DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100

SPM = {
    "t5": seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)
}

# MEENA_SPM_PATH = "/cns/yo-d/home/brain-meena/vocab/meena_0611.32000.model"
# MEENA_VOCABULARY = t5.data.SentencePieceVocabulary(MEENA_SPM_PATH)

VOCABULARY = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)

for kk in list(task_registry_ds['validation'].take(10)):
  kkk = kk['targets']
  print(kkk.shape)
  print(VOCABULARY.decode_tf(kkk))
