import sys
import os
sys.path.append(os.getcwd())
from final_exps.mar21_copy_length.utils import make_finetune_command


os.system(make_finetune_command('cnndm', 'jan23', 2, sys.argv[1]))

# python final_exps/mar21_copy_length/cnndm/jan23.py z && final_exps/mar21_copy_length/mtop/jan23.py z