from pstats import Stats
from tokenize import group
from webbrowser import get

from torch import layer_norm
from t5x.checkpoints import load_t5x_checkpoint
import sys
import os
import os.path as osp
import re
import jax 
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
import pandas as pd


mar4param_to_regex_dic = \
  {
    'q':'.*query',
    'k':'.*key',
    'v':'.*value',
    'o':'.*out',
    'pre_mlp_ln':'.*pre_mlp_layer_norm',
    'pre_self_attn_ln':"(encoder/.*pre_attention_layer)|(decoder/.*pre_self.*norm)",
    'pre_cross_attn_ln':'.*pre_cross',
    'enc_relpos':'enc.*relpos',
    'dec_relpos':'dec.*relpos',
    'after_dec_ln':'.*decoder_norm',
    'after_enc_ln':'.*encoder_norm',
    '':'ajhfadhfhgasdlfhasdfhdasbfasdfasdfadsfadsf',

  }


def init_with_custom_std(std, init_weights_dic, list_param_names):
  print(f"CCCCCCCCCCCCCUSTOM std {list_param_names} {std}\n"*10)
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for name in list_param_names:
    print('=====custom var======')
    print(name)
    regex = mar4param_to_regex_dic[name]
    print(regex)
    init_weights_dic = _init_PER_regex_params_with_custom_std(std, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_with_custom_value(value, init_weights_dic, list_param_names):
  print(f"CCCCCCCCCCCCCUSTOM VVVVVVValue {list_param_names} {value}\n"*10)
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for name in list_param_names:
    print('=====custom VALUE======')
    print(name)
    regex = mar4param_to_regex_dic[name]
    print(regex)
    init_weights_dic = _init_PER_regex_params_with_custom_value(value, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_with_per_param_mean_std_given_input_list_mar4_param_names(weights_dic_to_compute_stats, init_weights_dic, list_mar4_param_names):
  print("123456\n"*10)
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for name in list_mar4_param_names:
    print('=====MEAN AND STD======')
    print(name)
    regex = mar4param_to_regex_dic[name]
    print(regex)
    init_weights_dic = _init_PER_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_with_per_param_std_given_input_list_mar7_param_names(weights_dic_to_compute_stats, init_weights_dic, list_param_names):
  print("sd only\n"*10)
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for name in list_param_names:
    print('=====STD ONLY======')
    print(name)
    regex = mar4param_to_regex_dic[name]
    print(regex)
    init_weights_dic = _init_PER_regex_params_with_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_with_per_param_mean_given_input_list_param_names(weights_dic_to_compute_stats, init_weights_dic, list_param_names):
  print("MEAN MEAN MEAN only\n"*10)
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for name in list_param_names:
    print('=====MEAN ONLY======')
    print(name)
    regex = mar4param_to_regex_dic[name]
    print(regex)
    init_weights_dic = _init_PER_regex_params_with_mean(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic



####FINAL INIT EXPS CODE START#####

def compute_mean_and_std(weights):
  return np.mean(weights), np.std(weights)

def get_all_strings_who_match_regexp(regexp, strings):
  r = re.compile(regexp)
  return list(filter(r.match, strings))

def get_all_strings_who_match_regex(regexp, strings):
  r = re.compile(regexp)
  return list(filter(r.match, strings))


def return_per_param_grouping(param_names):
    grouping = [[x] for x in param_names]
    return grouping


def return_across_layers_grouping(param_names):
    encoder_layer_param_regexes = get_all_strings_who_match_regex('encoder.*layers_0.*', param_names)
    encoder_layer_param_regexes = [x.replace('0','.*') for x in encoder_layer_param_regexes]
    decoder_layer_param_regexes = get_all_strings_who_match_regex('decoder.*layers_0.*', param_names)
    decoder_layer_param_regexes = [x.replace('0','.*') for x in decoder_layer_param_regexes]
    non_layer_param_names = list(filter(lambda x:'layers' not in x, param_names))
    
    regexes = encoder_layer_param_regexes + decoder_layer_param_regexes + non_layer_param_names
    grouping = [get_all_strings_who_match_regex(regex, param_names) for regex in regexes]
    return grouping


def flatten_2d_list(lst):
    return [item for sublist in lst for item in sublist]


def return_per_layer_grouping(param_names, num_layers=6):
    grouping = []

    for e_or_d in ['encoder', 'decoder']:
        for i in range(num_layers):
            if e_or_d == 'encoder':
                grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_{i}/.*attention/', param_names))
            else:
                grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_{i}/encoder_decoder_attention/', param_names))
                grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_{i}/self_attention/', param_names))

            grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_{i}/.*mlp/', param_names))
            
    other_params = [[x] for x in (set(param_names) - set(flatten_2d_list(grouping)))]
    grouping = grouping + other_params
#     print(grouping)
    return grouping

def return_per_layer_across_layer_grouping(param_names):
    grouping = []

    for e_or_d in ['encoder', 'decoder']:
        grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_mlp_layer_norm/', param_names))
        if e_or_d == 'encoder':
            grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names))
            grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/', param_names))
        else:
            grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names))
            grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names))
  
            grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/', param_names))
            grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/', param_names))

        grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*mlp/', param_names))
        
    non_layer_param_names = list(filter(lambda x:'layers' not in x, param_names))
    grouping = grouping + [[x] for x in non_layer_param_names]
    
#     other_params = [[x] for x in (set(param_names) - set(flatten_list(grouping)))]
#     grouping = grouping + other_params
#     print(grouping)
    return grouping


def compute_combined_mean_and_std_given_list_of_param_names(param_names, weights_dic):
  result = []
  for name in param_names:
    result.append(weights_dic[name])
  concat_weights = np.concatenate(result, axis=None)
  return compute_mean_and_std(concat_weights)

#init_per_param_besides_all_norms
#all_pre_attn_norms_grouped_across_all_layers...MEAN
# pretrain base on LIME


def init_each_param_given_list_of_param_names_and_mean_std(param_names, weights_dic, mean, std):
  for param in param_names:
    if mean is None:
      weights_dic[param] = np.random.normal(loc=0.0, scale=std, size=weights_dic[param].shape)
    elif std is None:
      weights_dic[param] = np.ones(weights_dic[param].shape)*mean
    else:
      weights_dic[param] = np.random.normal(loc=mean, scale=std, size=weights_dic[param].shape)
  return weights_dic


def init_per_layer_across_layer_grouping(pretrain_weights_dic, init_weights_dic):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = return_per_layer_across_layer_grouping(list(pretrain_weights_dic.keys()))
  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n---------')

    if 'norm' in g[0]:
      print("norm!@!@#!@#!@#!@#!@#!@#!@#")
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, None)
    else:
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, None, std)
    print(g)
  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_per_layer_grouping(pretrain_weights_dic, init_weights_dic):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = return_per_layer_grouping(list(pretrain_weights_dic.keys()))
  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n---------')

    if 'norm' in g[0]:
      print("norm!@!@#!@#!@#!@#!@#!@#!@#")
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, None)
    else:
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, None, std)
    print(g)
  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic



def init_per_param_std_and_layer_norm_scale_mean(pretrain_weights_dic, init_weights_dic):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = return_per_param_grouping(list(pretrain_weights_dic.keys()))
  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n---------')

    if 'norm' in g[0]:
      print("norm!@!@#!@#!@#!@#!@#!@#!@#")
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, None)
    else:
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, None, std)
    print(g)
  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_param_std_across_layers_and_layer_norm_scale_mean_across_layers(pretrain_weights_dic, init_weights_dic):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = return_across_layers_grouping(list(pretrain_weights_dic.keys()))
  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n---------')

    if 'norm' in g[0]:
      print("norm!@!@#!@#!@#!@#!@#!@#!@#")
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, None)
    else:
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, None, std)
    print(g)
  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def return_only_attn_norm_all_together_grouping(param_names):
#     grouping = [[x] for x in param_names]
    group = list(filter(lambda name: ('attention' in name and 'norm' in name), param_names))
#     param_names_minus_group = sorted(list(set(param_names) - set(group)))
#     grouping = [group] + [[x] for x in param_names_minus_group]
#     print(grouping)

    grouping = [group]
    return grouping


def return_per_param_except_attn_norm_grouping(param_names):
    grouping = [[x] for x in param_names]
    grouping = list(filter(lambda x: not ('attention' in x[0] and 'norm' in x[0]), grouping))
    for g in grouping:
        assert len(g) == 1
    return grouping


def init_only_attn_norm_all_together_grouping(pretrain_weights_dic, init_weights_dic):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = return_only_attn_norm_all_together_grouping(list(pretrain_weights_dic.keys()))
  assert len(grouping) == 1

  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n---------')
    if 'norm' in g[0]:
      print("norm!@!@#!@#!@#!@#!@#!@#!@#")
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, None)
    else:
      assert False
    print(g)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_per_param_except_attn_norm_grouping(pretrain_weights_dic, init_weights_dic):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = return_per_param_except_attn_norm_grouping(list(pretrain_weights_dic.keys()))
  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n------------')
    print(g)
    assert len(g) == 1
    init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, std)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


INIT_ID_TO_INIT_FUNC_DIC = {'only_attn_norm_all_together': init_only_attn_norm_all_together_grouping, 'per_param_except_attn_norm':init_per_param_except_attn_norm_grouping}



####FINAL INIT EXPS CODE END#####


def concatenate_all_weights_whose_name_matches_regexp(regexp, weights_dic, check_match_count=-1, is_print=False):
  names_who_match_regexp = get_all_strings_who_match_regexp(regexp, weights_dic.keys())

  if check_match_count != -1:
    # print('=====')
    # print(regexp)
    # print(names_who_match_regexp)
    assert check_match_count == len(names_who_match_regexp)

  result = []
  for name in names_who_match_regexp:
    result.append(weights_dic[name])

  if is_print:
    print('=====')
    print(regexp)
    print(len(result))
    print(names_who_match_regexp)
  return np.concatenate(result, axis=None)


def load_weights_with_all_weights_statistics(weights_dic):
  weights_dic = traverse_util.flatten_dict(weights_dic, sep='/')

  init_weights_dic = {}
  all_weights = concatenate_all_weights_whose_name_matches_regexp('', weights_dic)
  all_weights_mean, all_weights_std = compute_mean_and_std(all_weights)
  for name, values in weights_dic.items():
    init_weights_dic[name] = np.random.normal(loc=all_weights_mean, scale=all_weights_std, size=values.shape)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def load_weights_with_layer_statistics(weights_dic):
  weights_dic = traverse_util.flatten_dict(weights_dic, sep='/')
  init_weights_dic = {}

  for name, values in weights_dic.items():
    values_mean, values_std = compute_mean_and_std(values)
    init_weights_dic[name] = np.random.normal(loc=values_mean, scale=values_std, size=values.shape)

  for enc_or_dec in ['encoder', 'decoder']:
    for i in range(12):
      regex = f'.*{enc_or_dec}.*layers_{i}/.*'
      regex_params = get_all_strings_who_match_regexp(regex, weights_dic.keys())
      assert len(regex_params) > 4
      regex_mean, regex_var = compute_mean_and_std(concatenate_all_weights_whose_name_matches_regexp(regex, weights_dic))
      for param in regex_params:
        init_weights_dic[param] = np.random.normal(loc=regex_mean, scale=regex_var, size=weights_dic[param].shape)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


# def load_regex_weights_with_stats(regex, weights_to_maybe_load, init_weights):
#   weights_to_maybe_load = traverse_util.flatten_dict(weights_to_maybe_load, sep='/')

#   names_of_weights_to_load = get_all_strings_who_match_regexp(regex, weights_to_maybe_load.keys())
#   for n in names_of_weights_to_load:
#     init_weights = 

#   init_weights = traverse_util.unflatten_dict(init_weights, sep='/')
#   return init_weights

# def load_regex_weights(regex, weights_to_load, init_weights):
  

########

def load_with_per_param_stats_excluding_relpos_weights(weights_to_maybe_load, init_weights):
  return load_excluding_regex_weights('.*relpos.*', weights_to_maybe_load, init_weights)


def load_with_per_param_stats_excluding_regex_weights(regex_to_exclude, weights_to_maybe_load, init_weights):
  weights_to_maybe_load = traverse_util.flatten_dict(weights_to_maybe_load, sep='/')
  init_weights = traverse_util.flatten_dict(init_weights, sep='/')

  names_of_weights_to_exclude = get_all_strings_who_match_regexp(regex_to_exclude, weights_to_maybe_load.keys())
  if regex_to_exclude == '.*relpos.*':
    assert len(names_of_weights_to_exclude) == 2
  for name, weights in weights_to_maybe_load.items():
    if name in names_of_weights_to_exclude:
      continue
    weights_mean, weights_std = compute_mean_and_std(weights)
    init_weights[name] = np.random.normal(loc=weights_mean, scale=weights_std, size=weights.shape)

  init_weights = traverse_util.unflatten_dict(init_weights, sep='/')
  return init_weights


def load_all_excluding_relpos_weights(weights_to_maybe_load, init_weights):
  return load_excluding_regex_weights('.*relpos.*', weights_to_maybe_load, init_weights)


def load_excluding_regex_weights(regex_to_exclude, weights_to_maybe_load, init_weights):
  weights_to_maybe_load = traverse_util.flatten_dict(weights_to_maybe_load, sep='/')
  init_weights = traverse_util.flatten_dict(init_weights, sep='/')

  names_of_weights_to_exclude = get_all_strings_who_match_regexp(regex_to_exclude, weights_to_maybe_load.keys())
  if regex_to_exclude == '.*relpos.*':
    assert len(names_of_weights_to_exclude) == 2
  for name, weights in weights_to_maybe_load.items():
    if name in names_of_weights_to_exclude:
      continue
    init_weights[name] = weights

  init_weights = traverse_util.unflatten_dict(init_weights, sep='/')
  return init_weights

#######


def init_weights_with_weights_stats(weights_dic):
  weights_dic = traverse_util.flatten_dict(weights_dic, sep='/')
  init_weights_dic = {}

  for name, values in weights_dic.items():
    values_mean, values_std = compute_mean_and_std(values)
    init_weights_dic[name] = np.random.normal(loc=values_mean, scale=values_std, size=values.shape)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def only_load_relpos_embed(weights_dic_with_relpos_embed_to_load, init_weights_dic):
  weights_dic_with_relpos_embed_to_load = traverse_util.flatten_dict(weights_dic_with_relpos_embed_to_load, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for name, values in weights_dic_with_relpos_embed_to_load.items():
    if 'relpos' in name:
      init_weights_dic[name] = values

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def only_load_relpos_embed_with_statistics(weights_dic_with_relpos_embed_to_load, init_weights_dic):
  weights_dic_with_relpos_embed_to_load = traverse_util.flatten_dict(weights_dic_with_relpos_embed_to_load, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  
  for name, values in weights_dic_with_relpos_embed_to_load.items():
    if 'relpos' in name:
      values_mean, values_std = compute_mean_and_std(values)
      init_weights_dic[name] = np.random.normal(loc=values_mean, scale=values_std, size=values.shape)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def _init_regex_params_with_std(weights_dic_to_compute_stats, init_weights_dic, regex):
  paramss = get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  concat_weights = concatenate_all_weights_whose_name_matches_regexp(regex, weights_dic_to_compute_stats, check_match_count=len(paramss))
  _, std = compute_mean_and_std(concat_weights)
  for param in paramss:
    if 'norm' in param:
      raise NotImplementedError
    else:
      init_weights_dic[param] = np.random.normal(loc=0.0, scale=std, size=init_weights_dic[param].shape)

  return init_weights_dic

def _init_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex):
  paramss = get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  concat_weights = concatenate_all_weights_whose_name_matches_regexp(regex, weights_dic_to_compute_stats, check_match_count=len(paramss))
  mean, std = compute_mean_and_std(concat_weights)
  for param in paramss:
    init_weights_dic[param] = np.random.normal(loc=mean, scale=std, size=init_weights_dic[param].shape)

  return init_weights_dic

def _init_PER_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex):
  if regex == mar4param_to_regex_dic['']:
    return init_weights_dic
  paramss = get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())
  print(paramss)
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  for param in paramss:
    mean, std = compute_mean_and_std(weights_dic_to_compute_stats[param])
    init_weights_dic[param] = np.random.normal(loc=mean, scale=std, size=init_weights_dic[param].shape)
    assert std != 0
    assert np.mean(init_weights_dic[param]) != 0
  return init_weights_dic


def _init_PER_regex_params_with_std(weights_dic_to_compute_stats, init_weights_dic, regex):
  if regex == mar4param_to_regex_dic['']:
    return init_weights_dic
  paramss = get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  print(paramss)
  for param in paramss:
    _, std = compute_mean_and_std(weights_dic_to_compute_stats[param])
    if 'norm' in param:
      init_weights_dic[param] = np.random.normal(loc=1.0, scale=std, size=init_weights_dic[param].shape)
    else:
      init_weights_dic[param] = np.random.normal(loc=0.0, scale=std, size=init_weights_dic[param].shape)
    assert std != 0
    assert np.mean(init_weights_dic[param]) != 0
  return init_weights_dic


def _init_PER_regex_params_with_mean(weights_dic_to_compute_stats, init_weights_dic, regex):
  if regex == mar4param_to_regex_dic['']:
    return init_weights_dic
  paramss = get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  print(paramss)
  for param in paramss:
    mean, _ = compute_mean_and_std(weights_dic_to_compute_stats[param])
    init_weights_dic[param] = np.ones(init_weights_dic[param].shape) * mean

    # if 'norm' in param:
    #   init_weights_dic[param] = np.ones(init_weights_dic[param].shape) * mean
    # else:
    #   init_weights_dic[param] = np.random.normal(loc=mean, scale=np.std(init_weights_dic[param]), size=init_weights_dic[param].shape)
    # assert std != 0

    # assert np.mean(init_weights_dic[param]) != 0
  return init_weights_dic



def _init_PER_regex_params_with_custom_value(value, init_weights_dic, regex):
  if regex == mar4param_to_regex_dic['']:
    return init_weights_dic
  paramss = get_all_strings_who_match_regexp(regex, init_weights_dic.keys())
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  print(paramss)
  for param in paramss:
    init_weights_dic[param] = np.ones(init_weights_dic[param].shape) * value

    # if 'norm' in param:
    #   init_weights_dic[param] = np.ones(init_weights_dic[param].shape) * mean
    # else:
    #   init_weights_dic[param] = np.random.normal(loc=mean, scale=np.std(init_weights_dic[param]), size=init_weights_dic[param].shape)
    # assert std != 0

    # assert np.mean(init_weights_dic[param]) != 0
  return init_weights_dic


def _init_PER_regex_params_with_custom_std(std, init_weights_dic, regex):
  if regex == mar4param_to_regex_dic['']:
    return init_weights_dic
  paramss = get_all_strings_who_match_regexp(regex, init_weights_dic.keys())
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  print(paramss)
  for param in paramss:
    if 'norm' in param:
      raise NotImplementedError()
    else:
      init_weights_dic[param] = np.random.normal(loc=0.0, scale=std, size=init_weights_dic[param].shape)
  return init_weights_dic


def _init_PER_regex_params_with_mean_and_from_scratch_std(weights_dic_to_compute_stats, init_weights_dic, regex):
  paramss = get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())
  assert type(paramss[0]) == str
  assert len(paramss) > 0
  for param in paramss:
    mean, _ = compute_mean_and_std(weights_dic_to_compute_stats[param])
    if 'norm' in param:
      init_weights_dic[param] = mean * np.ones(init_weights_dic[param].shape)
    else:
      scale = np.std(init_weights_dic[param])
      init_weights_dic[param] = np.random.normal(loc=mean, scale=scale, size=init_weights_dic[param].shape)
    assert np.mean(init_weights_dic[param]) != 0
  return init_weights_dic


#############
# per layer attention var
#############
def init_with_per_layer_attention_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for i in range(12):
    # encoder per layer self attn
    regexess = [f'encoder.*layers_{i}/.*attention/', f'decoder.*layers_{i}/.*self_attention/', f'decoder.*layers_{i}/.*encoder_decoder_attention/']
    for regex in regexess:
      # print(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys()))
      assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 4
      init_weights_dic = _init_regex_params_with_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic

  
#############
# is_init_with_relpos_mean_std
# is_init_with_token_mean_std
# is_init_with_per_layer_attention_mean_std
# is_init_with_per_QKVO_mean_std
#############


def init_with_token_mean_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = "token"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 1
  init_weights_dic = _init_PER_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic

def init_with_relpos_mean_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = ".*relpos.*"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 2
  init_weights_dic = _init_PER_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic

def init_with_per_layer_attention_mean_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  for i in range(12):
    # encoder per layer self attn
    regexess = [f'encoder.*layers_{i}/.*attention/', f'decoder.*layers_{i}/.*self_attention/', f'decoder.*layers_{i}/.*encoder_decoder_attention/']
    for regex in regexess:
      # print(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys()))
      assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 4
      init_weights_dic = _init_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_with_per_QKVO_mean_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = ".*attention/.*((query)|(key)|(value)|(out)).*"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 144
  init_weights_dic = _init_PER_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_with_per_mlp_mean_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = ".*mlp/.*"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 48
  init_weights_dic = _init_PER_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic

def init_with_per_layer_norm_mean_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = ".*norm/.*"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 62
  init_weights_dic = _init_PER_regex_params_with_mean_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


### std only ###

def init_with_per_layer_norm_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = ".*norm/.*"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 62
  init_weights_dic = _init_PER_regex_params_with_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_with_per_QKVO_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = ".*attention/.*((query)|(key)|(value)|(out)).*"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 144
  init_weights_dic = _init_PER_regex_params_with_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic

def init_with_relpos_std(weights_dic_to_compute_stats, init_weights_dic):
  weights_dic_to_compute_stats = traverse_util.flatten_dict(weights_dic_to_compute_stats, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  regex = ".*relpos.*"
  assert len(get_all_strings_who_match_regexp(regex, weights_dic_to_compute_stats.keys())) == 2
  init_weights_dic = _init_PER_regex_params_with_std(weights_dic_to_compute_stats, init_weights_dic, regex)

  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic




if __name__=='__main__':
  # ckpt_path = '/mnt/disks/persist/t5_training_models/synthetic_tasks/1-24/pretrain/0_1/checkpoint_80000'
  ckpt_path = '/mnt/disks/persist/t5_training_models/CHECKPOINTS_FROM_GCS/synthetic_tasks/1-24/pretrain/0_1/checkpoint_40000'
  ckpt = load_t5x_checkpoint(ckpt_path)

  # random_init_ckpt_path = '/mnt/disks/persist/t5_training_models/synthetic_tasks/1-30/finetune/random_init/checkpoint_1'
  # random_init_ckpt = load_t5x_checkpoint(random_init_ckpt_path)

  weights_dic = ckpt['target']
  weights_dic = traverse_util.flatten_dict(weights_dic, sep='/')
  init_weights_dic = {k:np.zeros(v.shape) for k,v in weights_dic.items()}
  init_weights_dic = init_with_per_layer_attention_std(weights_dic, init_weights_dic)
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')
  for k, v in init_weights_dic.items():
    # assert np.all(weights_dic[k] != v)
    if 'attention/' in k:
      assert np.mean(init_weights_dic[k]) != 0
    if 'attention/' not in k:
      assert np.mean(init_weights_dic[k]) == 0
    print('\n\n---')
    print(f'==={k}===')
    print('=means=')
    print(np.mean(weights_dic[k]))
    print(np.mean(init_weights_dic[k]))
    print('=stds=')
    print(np.std(weights_dic[k]))
    print(np.std(init_weights_dic[k]))
