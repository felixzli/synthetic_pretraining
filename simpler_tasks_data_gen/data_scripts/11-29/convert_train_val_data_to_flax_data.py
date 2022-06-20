import glob
import os
import os.path as osp


def get_second_index(input_string, sub_string):
    return input_string.index(sub_string, input_string.index(sub_string) + 1)


train_files, valid_files = sorted(glob.glob('./data/11-29/*/train*')), sorted(glob.glob('./data/11-29/*/valid*'))

for t, v in zip(train_files, valid_files):
  
  tbn = osp.basename(t)
  vbn = osp.basename(v)

  num_train = tbn[5:tbn.index('_')]
  num_valid = vbn[5:vbn.index('_')]
  data_config_str = tbn[get_second_index(tbn, '_')+1:tbn.rindex('_')]

  flax_data_dir = osp.join('flax_data/11-29/', f'{data_config_str}_{num_train}')
  os.makedirs(flax_data_dir, exist_ok=True)
  
  datas = (t,v)
  srcs = (osp.join(flax_data_dir, 'train.src'), osp.join(flax_data_dir, 'valid.src'))
  tgts = (osp.join(flax_data_dir, 'train.tgt'), osp.join(flax_data_dir, 'valid.tgt'))
  for data, src, tgt in zip(datas, srcs, tgts):
    os.system(f'python data_scripts/convert_to_flax_data.py \
              --data_path {data} \
              --new_flax_data_src_path {src} \
              --new_flax_data_tgt_path {tgt}')