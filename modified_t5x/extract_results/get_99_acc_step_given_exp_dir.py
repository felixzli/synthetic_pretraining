from extract_info_from_tfevents import get_first_99_acc_and_step_num
import glob
import sys
import os.path as osp

if __name__ == '__main__':
  exp_dir = sys.argv[1]
  # example: /mnt/disks/persist/t5_training_models/final_exps/mar21_cfg/deduct_pretrain_t5small
  assert osp.isdir(exp_dir)
  
  # if len(glob.glob(osp.join(exp_dir, 'training_eval/*','events*'))) > 1:
  #   tfevents_path = glob.glob(osp.join(exp_dir, 'training_eval/1M_1_23_lime/','events*'))[0]
  # else:
    # assert len(glob.glob(osp.join(exp_dir, 'training_eval/*','events*'))) == 1
  tfevents_paths = glob.glob(osp.join(exp_dir, 'training_eval/*','events*'))
  # example: /mnt/disks/persist/t5_training_models/final_exps/mar21_cfg/deduct_pretrain_t5small/training_eval/mar21_cfg_deduct/events.out.tfevents.1648094780.t1v-n-fd8d27d0-w-0.194310.1.v2
  for tfevents_path in tfevents_paths:
    get_first_99_acc_and_step_num(tfevents_path, print_result=True, print_all_accs_and_step_nums=False)
