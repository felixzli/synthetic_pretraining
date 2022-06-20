import sys
import os
sys.path.append(os.getcwd())
from final_exps.mar25_copy_ablate_vocab.utils import load_all_except_embed_make_finetune_command

if __name__ == '__main__':
  if len(sys.argv) == 1:
    sys.argv.append(-1)
  for vocab_size in [2, 1000, 10000]:
    os.system(load_all_except_embed_make_finetune_command('mtop', vocab_size, 252, 252, 3, sys.argv[1]))