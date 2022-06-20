import sys
import os
sys.path.append(os.getcwd())
from final_exps.mar21_copy_length.utils import make_finetune_command

if __name__ == '__main__':
  if len(sys.argv) == 1:
    sys.argv.append(-1)

  for length in [28, 252, 1020]:
    os.system(make_finetune_command('cnndm', length, 2, sys.argv[1]))