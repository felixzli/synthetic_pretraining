
import sys
import os
import os.path as osp
import re
import jax 
import jax.numpy as jnp
from flax import traverse_util
import numpy as np


def filter_for_nonlayer_params(params, include_token_embed=True):
  if include_token_embed:
    return filter(lambda x:'layers' not in x, params)
  else:
    return filter(lambda x:'layers' not in x and 'token_embedder/embedding' not in x, params)


def filter_for_NOT_regex(params, regex):
  # return every param that does NOT match regex
  match_regex = set(get_all_strings_who_match_regex(regex, params))
  result = []
  for p in params:
    if p not in match_regex:
      result.append(p)
  return result

def get_all_strings_who_DONT_match_regex(regex, strings):
  return filter_for_NOT_regex(strings, regex)

def get_all_strings_who_match_regex(regexp, strings):
  r = re.compile(regexp)
  return list(filter(r.match, strings))


def get_per_param_grouping(params):
  return [[x] for x in (set(params) - set(['token_embedder/embedding']))]


def get_layer_params_per_param_grouping(params):
  print('foo')
  non_layer_params = set(filter_for_nonlayer_params(params))
  params = list(set(params) - non_layer_params)
  params = sorted(params)
  return [[x] for x in params]


def get_layer_params_across_layer_grouping(params):
  encoder_layer_param_regexes = get_all_strings_who_match_regex('encoder.*layers_0.*', params)
  encoder_layer_param_regexes = [x.replace('0','.*') for x in encoder_layer_param_regexes]
  decoder_layer_param_regexes = get_all_strings_who_match_regex('decoder.*layers_0.*', params)
  decoder_layer_param_regexes = [x.replace('0','.*') for x in decoder_layer_param_regexes]
  
  regexes = encoder_layer_param_regexes + decoder_layer_param_regexes 
  grouping = [get_all_strings_who_match_regex(regex, params) for regex in regexes]
  return grouping


# def get_across_layer_grouping(params):
#   encoder_layer_param_regexes = get_all_strings_who_match_regex('encoder.*layers_0.*', params)
#   encoder_layer_param_regexes = [x.replace('0','.*') for x in encoder_layer_param_regexes]
#   decoder_layer_param_regexes = get_all_strings_who_match_regex('decoder.*layers_0.*', params)
#   decoder_layer_param_regexes = [x.replace('0','.*') for x in decoder_layer_param_regexes]
  
#   regexes = encoder_layer_param_regexes + decoder_layer_param_regexes 
#   grouping = [get_all_strings_who_match_regex(regex, params) for regex in regexes]
#   grouping = grouping + [[x] for x in filter_for_nonlayer_params(params)]

#   return grouping


def get_layer_params_across_layer_big_grouping(param_names):
  grouping = []

  for e_or_d in ['encoder', 'decoder']:
    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_mlp_layer_norm/', param_names))
    if e_or_d == 'encoder':
      grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names))
      grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/', param_names))
    else:
      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names) + get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/', param_names) + get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/', param_names)
      grouping.append(group)

    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*mlp/', param_names))
      
  grouping = grouping 
  
  return grouping


def get_across_layer_big_grouping(param_names):
  grouping = []

  for e_or_d in ['encoder', 'decoder']:
    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_mlp_layer_norm/', param_names))
    if e_or_d == 'encoder':
      grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names))
      grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/', param_names))
    else:
      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names) + get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/', param_names) + get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/', param_names)
      grouping.append(group)

    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*mlp/', param_names))
  
  grouping = grouping + [[x] for x in filter_for_nonlayer_params(param_names)]

  return grouping


def get_layer_params_across_layer_big_big_grouping(param_names):
  grouping = []

  for e_or_d in ['encoder', 'decoder']:
    mlp_attn_group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*mlp/', param_names)
    ln_group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_mlp_layer_norm/', param_names)
    if e_or_d == 'encoder':
      ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names)

      mlp_attn_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/', param_names)
    else:
      ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names) 
      ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names)

      mlp_attn_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/', param_names) 
      mlp_attn_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/', param_names)

    grouping.append(ln_group)
    grouping.append(mlp_attn_group)

  grouping = grouping 
  
  return grouping


# def get_mlp_across_layer_grouping(param_names):
#   grouping = []

#   for e_or_d in ['encoder', 'decoder']:
#     mlp_attn_group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*mlp/', param_names)
#     grouping.append(mlp_attn_group)
  
#   return grouping


# def get_attention_across_layer_grouping(param_names):
#   grouping = []

#   for e_or_d in ['encoder', 'decoder']:
#     mlp_attn_group = []
#     if e_or_d == 'encoder':

#       mlp_attn_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/', param_names)
#     else:
#       mlp_attn_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/', param_names) 
#       mlp_attn_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/', param_names)

#     grouping.append(mlp_attn_group)


#   return grouping


# def get_premlpln_across_layer_grouping(param_names):
#   grouping = []

#   for e_or_d in ['encoder', 'decoder']:
#     ln_group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_mlp_layer_norm/', param_names)
#     grouping.append(ln_group)
  
#   return grouping


# def get_preattnln_across_layer_grouping(param_names):
#   grouping = []

#   for e_or_d in ['encoder', 'decoder']:
#     ln_group = []
#     if e_or_d == 'encoder':
#       ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names)

#     else:
#       ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names) 
#       ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names)

#     grouping.append(ln_group)

#   grouping = grouping 
  
#   return grouping



def get_layer_params_per_param_grouping_exclude_preattnln(params):
  non_layer_params = set(filter_for_nonlayer_params(params))
  preattnln_params = set(filter(lambda x: ('attention' in x and 'norm' in x), params))
  params = list(set(params) - non_layer_params - preattnln_params)
  params = sorted(params)
  return [[x] for x in params]


def compute_mean_and_std(weights):
  return np.mean(weights), np.std(weights)


def compute_combined_mean_and_std_given_list_of_param_names(param_names, weights_dic):
  result = []
  for name in param_names:
    result.append(weights_dic[name])
  concat_weights = np.concatenate(result, axis=None)
  assert len(concat_weights.shape) == 1
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


def init_scale(pretrain_weights_dic, init_weights_dic, grouping_fn):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = grouping_fn(list(pretrain_weights_dic.keys()))
  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n---------')
    print(g)
    if 'norm' in g[0]:
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, None)
      print(f'mean={mean}q')
    else:
      init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, None, std)
      print(f'std={std}')


  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def init_mean_std(pretrain_weights_dic, init_weights_dic, grouping_fn):
  pretrain_weights_dic = traverse_util.flatten_dict(pretrain_weights_dic, sep='/')
  init_weights_dic = traverse_util.flatten_dict(init_weights_dic, sep='/')

  grouping = grouping_fn(list(pretrain_weights_dic.keys()))
  for g in grouping:
    mean, std = compute_combined_mean_and_std_given_list_of_param_names(g, pretrain_weights_dic)
    print('\n---------')
    init_weights_dic = init_each_param_given_list_of_param_names_and_mean_std(g, init_weights_dic, mean, std)
    print(g)
    print(f'mean={mean} ____ std={std}')
  init_weights_dic = traverse_util.unflatten_dict(init_weights_dic, sep='/')
  return init_weights_dic


def flatten_2d_list(lst):
  for l in lst:
    assert type(l) == list
  return [item for sublist in lst for item in sublist]


def get_nonlayer_ln_per_param_grouping(params):
  non_layer_params = list(filter_for_nonlayer_params(params))
  grouping = []
  for p in non_layer_params:
    if 'norm' in p:
      grouping.append([p])
  return grouping


def get_relpos_per_param_grouping(params):
  non_layer_params = list(filter_for_nonlayer_params(params))
  grouping = []
  for p in non_layer_params:
    if 'relpos' in p:
      grouping.append([p])
  return grouping


def get_nonlayer_params_per_param_grouping(params):
  non_layer_params = list(filter_for_nonlayer_params(params))
  grouping = [[x] for x in non_layer_params]
  return grouping


def get_token_embed_per_param_grouping(params):
  non_layer_params = list(filter_for_nonlayer_params(params))
  grouping = []
  for p in non_layer_params:
    if 'token' in p:
      grouping.append([p])
  return grouping

def get_across_layer_big_grouping_v2(param_names):
  grouping = []

  for e_or_d in ['encoder', 'decoder']:
    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_mlp_layer_norm/', param_names))
    if e_or_d == 'encoder':
      grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names))
      # grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/', param_names))

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/query', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/key', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/value', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/out', param_names)
      grouping.append(group)
    else:
      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names) + get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/query', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/key', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/query', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/key', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/value', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/out', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/value', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/out', param_names)
      grouping.append(group)


    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*mlp/', param_names))
  
  grouping = grouping + [[x] for x in filter_for_nonlayer_params(param_names)]

  return grouping

def get_layer_params_across_layer_big_grouping_v2(param_names):
  grouping = []

  for e_or_d in ['encoder', 'decoder']:
    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_mlp_layer_norm/', param_names))
    if e_or_d == 'encoder':
      grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names))
      # grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/', param_names))

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/query', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/key', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/value', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*attention/out', param_names)
      grouping.append(group)
    else:
      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names) + get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/query', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/key', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/query', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/key', param_names)
      grouping.append(group)

      group = get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/value', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/encoder_decoder_attention/out', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/value', param_names)
      group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/self_attention/out', param_names)
      grouping.append(group)


    grouping.append(get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/.*mlp/', param_names))
  

  return grouping


def get_nonlayer_ln_and_relpos_per_param_grouping(params):
  non_layer_params = list(filter_for_nonlayer_params(params))
  grouping = []
  for p in non_layer_params:
    if 'relpos' in p or 'norm' in p:
      grouping.append([p])
  return grouping

def get_per_param_grouping_exclude_preattnln(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  preattnln_params = set(filter(lambda x: ('attention' in x and 'norm' in x), params))
  params = list(set(params) - preattnln_params)
  params = sorted(params)
  return [[x] for x in params]


def get_preattnln_per_param_grouping(params):
  params = sorted(get_all_strings_who_match_regex('.*attention.*norm', params))
  return [[x] for x in params]


def get_per_param_grouping_exclude_relpos_and_preattnln(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  relpos_and_preattnln_params = set(filter(lambda x: ('attention' in x and 'norm' in x) or 'relpos' in x, params))
  params = list(set(params) - relpos_and_preattnln_params)
  params = sorted(params)
  return [[x] for x in params]


def get_per_param_grouping_exclude_relpos(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  relpos_params = set(filter(lambda x: 'relpos' in x, params))
  params = list(set(params) - relpos_params)
  params = sorted(params)
  return [[x] for x in params]


def get_per_param_grouping_exclude_premlpln(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  
  params = sorted(get_all_strings_who_DONT_match_regex('.*mlp.*norm', params))
  return [[x] for x in params]


def get_premlpln_per_param_grouping(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  
  params = sorted(get_all_strings_who_match_regex('.*mlp.*norm', params))
  return [[x] for x in params]


def get_per_param_grouping_exclude_qkvo(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  
  params = sorted(get_all_strings_who_DONT_match_regex('.*attention/', params))
  return [[x] for x in params]


def get_qkvo_per_param_grouping(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  
  params = sorted(get_all_strings_who_match_regex('.*attention/', params))
  return [[x] for x in params]


def get_per_param_grouping_exclude_mlp(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  
  params = sorted(get_all_strings_who_DONT_match_regex('.*mlp/', params))
  return [[x] for x in params]


def get_mlp_per_param_grouping(params):
  # non_layer_params = set(filter_for_nonlayer_params(params))
  
  params = sorted(get_all_strings_who_match_regex('.*mlp/', params))
  return [[x] for x in params]


def get_relpos_together_grouping(params):
  relpos_params = list(filter(lambda x: 'relpos' in x, params))
  return [relpos_params]


def get_relpos_together_and_preattnln_across_layer_grouping(params):
  param_names = params
  grouping = []

  for e_or_d in ['encoder', 'decoder']:
    ln_group = []
    if e_or_d == 'encoder':
      ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_attention_layer_norm/', param_names)

    else:
      ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_self_attention_layer_norm/', param_names) 
      ln_group += get_all_strings_who_match_regex(f'{e_or_d}/layers_.*/pre_cross_attention_layer_norm/', param_names)

    grouping.append(ln_group)

  relpos_params = list(filter(lambda x: 'relpos' in x, params))
  grouping.append(relpos_params)
  return grouping


def get_across_ALL_layer_grouping(params):
  grouping = []
  print('.')
  print(f"total length params {len(params)}")
  print('.')

  group = get_all_strings_who_match_regex(f'.*query', params)
  grouping.append(group)

  group = get_all_strings_who_match_regex(f'.*key', params)
  grouping.append(group)

  group = get_all_strings_who_match_regex(f'.*value', params)
  grouping.append(group)

  group = get_all_strings_who_match_regex(f'.*attention.*out', params)
  grouping.append(group)

  grouping.append(get_all_strings_who_match_regex(f'.*mlp/', params))

  grouping.append(get_all_strings_who_match_regex('.*pre_mlp_layer_norm', params))

  grouping.append(get_all_strings_who_match_regex('.*attention_layer_norm', params))

  params_set = set(params)
  nonlayer_params_grouping = [['encoder/encoder_norm/scale', 'decoder/decoder_norm/scale'],
                              ['decoder/relpos_bias/rel_embedding', 'encoder/relpos_bias/rel_embedding']]
  flattened_nonlayer_params_grouping = flatten_2d_list(nonlayer_params_grouping)
  for p in flattened_nonlayer_params_grouping:
    print(p)
    assert p in params_set

  grouping = grouping + nonlayer_params_grouping

  grouping_num_params = sum([len(x) for x in grouping])
  print('.')
  print(f'grouping_num_params {grouping_num_params}')
  print('.')
  assert grouping_num_params == len(params) - 1

  return grouping


def get_across_ALL_layer_AND_per_layer_grouping(params):
  grouping = []
  print('.')
  print(f"total length params {len(params)}")
  print('.')

  gs = get_all_strings_who_match_regex
  layer_prefix = f''
  
  group = gs(f'{layer_prefix}.*attention/', params) + gs(f'{layer_prefix}.*mlp/', params)
  grouping.append(group)

  # group = gs(f'{layer_prefix}.*pre_mlp_layer_norm', params) + gs(f'{layer_prefix}.*attention_layer_norm', params)
  group = gs('.*norm', params)
  grouping.append(group)
  
  # non_layer_params = list(filter_for_nonlayer_params(params))
  # non_layer_params_nofilter_for_NOT_regex('.*norm', non_layer_params)

  # rest_params_grouping = [gs('.*relpos', params)]
  grouping = grouping + [gs('.*relpos', params)]

  grouping_num_params = sum([len(x) for x in grouping])
  print('.')
  print(f'grouping_num_params {grouping_num_params}')
  print('.')
  assert grouping_num_params == len(params) - 1

  return grouping

# preattnln_across_ALL_layer_grouping__init_scale
def get_preattnln_across_ALL_layer_grouping(params):
  grouping = []
  print('.')
  print(f"total length params {len(params)}")
  print('.')

  grouping.append(get_all_strings_who_match_regex('.*attention_layer_norm', params))


  grouping_num_params = sum([len(x) for x in grouping])
  print('.')
  print(f'grouping_num_params {grouping_num_params}')
  print('.')
  # assert grouping_num_params == len(params)

  return grouping


def get_per_layer_grouping(params):
  grouping = []
  print('.')
  print(f"total length params {len(params)}")
  print('.')

  gs = get_all_strings_who_match_regex
  for i in range(6):
    for e_or_d in ['encoder', 'decoder']:
      layer_prefix = f'{e_or_d}/layers_{i}/'
      
      group = gs(f'{layer_prefix}.*attention/', params) + gs(f'{layer_prefix}.*mlp/', params)
      grouping.append(group)

      group = gs(f'{layer_prefix}.*pre_mlp_layer_norm', params) + gs(f'{layer_prefix}.*attention_layer_norm', params)
      grouping.append(group)
    
  nonlayer_params_grouping = [[x] for x in filter_for_nonlayer_params(params, include_token_embed=False)]
  grouping = grouping + nonlayer_params_grouping

  grouping_num_params = sum([len(x) for x in grouping])
  print('.')
  print(f'grouping_num_params {grouping_num_params}')
  print('.')
  assert grouping_num_params == len(params) - 1

  return grouping

if __name__ == '__main__':
  pass
  # print(param_names)
  # print(len(param_names))
  # print(len(old_param_names))
