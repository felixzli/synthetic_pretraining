import seqio
import os
import os.path as osp
from tasks import TaskRegistry
import sys
# tasks = ['cnn_dailymail_v002', 'nonsense_summary_tasks','cnndm_from_pretraining_with_nonsense_paper']
# task = tasks[int(sys.argv[1])]
# task = 'copy_1_11_t5_token_basic_unary_tasks'
task = sys.argv[1]
task_registry_ds = {}
for split in ['validation']:
  task_registry_ds[split] = TaskRegistry.get_dataset(task, sequence_length=None,
                                                    split=split, shuffle=False)

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100

SPM = {
    "t5": seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)
}

VOCABULARY = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)
os.makedirs('tmp', exist_ok=True)
with open(f'tmp/{task}_take10.txt', mode='w') as f:
  for data in list(task_registry_ds['validation'].take(10)):
    # data.keys() ---> dict_keys(['inputs_pretokenized', 'inputs', 'targets_pretokenized', 'targets'])
    # inp_pre = data['inputs_pretokenized'].numpy().decode('utf-8')
    inp = data['inputs'] 
    decoded_inp = VOCABULARY.decode_tf(inp).numpy().decode('utf-8')
    # tgt_pre = data['targets_pretokenized'].numpy().decode('utf-8')
    tgt = data['targets'] 
    decoded_tgt = VOCABULARY.decode_tf(tgt).numpy().decode('utf-8')

    print('\n')
    # print('======== input_pretokenized')
    # print(inp_pre)
    print('======== VOCABULARY.decode_tf(tokenized_input)')
    print(decoded_inp)

    # print('======== target_pretokenized')
    # print(tgt_pre)
    print('======== VOCABULARY.decode_tf(tokenized_target)')
    print(decoded_tgt)

    f.writelines([ 
                  # '======== input_pretokenized\n', inp_pre + '\n',
                '======== VOCABULARY.decode_tf(input)\n', decoded_inp + '\n',
                # '======== target_pretokenized\n', tgt_pre + '\n',
                  '======== VOCABULARY.decode_tf(target)\n', decoded_tgt + '\n', '\n'
                  ])
