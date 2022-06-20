from time import time
from numpy import random
from nltk import CFG
import random
import numpy as np
from nltk.parse import RecursiveDescentParser
from parso import parse
from cfg_utils import get_terms_lst, get_vars_lst, ARROW_SYMBOL, sample_cfg_rules, sample_var_term_to_tok_dic
from generate_specific_depth_sents_and_derivs import generate_specific_depth_sents_and_derivs, convert_deriv_to_parse_tree_string
from timeout import timeout, TimeoutError



def convert_rules_to_token_lst(rules:list, special_tokens, var_term_to_tok_dic):    
  assert len(special_tokens) >= 2
  arrow_tok = special_tokens[0]
  rule_divider_tok = special_tokens[1]

  token_lst = []
  for r in rules:
    r = r.split(' ')
    assert len(r) >= 3

    token_lst.append(var_term_to_tok_dic[r[0]])

    assert r[1] == ARROW_SYMBOL
    token_lst.append(arrow_tok)

    rhs_symbols = r[2:]
    for symbol in rhs_symbols:
      if "'" in symbol:
        symbol = eval(symbol)
      token_lst.append(var_term_to_tok_dic[symbol])
      
    token_lst.append(rule_divider_tok)

  return token_lst[:-1] # because last token is rule_divider_tok


def convert_sent_to_token_lst(sent:list, var_term_to_tok_dic):    
  token_lst = []
  for term in sent:
    token_lst.append(var_term_to_tok_dic[term])
  return token_lst

 
def convert_sentss_to_token_lst(sentss:list, special_tokens, var_term_to_tok_dic):
  sent_divider_tok = special_tokens[0]
  token_lst = []
  for sent in sentss:
    sent_toks = convert_sent_to_token_lst(sent, var_term_to_tok_dic)
    token_lst.extend(sent_toks)
    token_lst.append(sent_divider_tok)
  return token_lst[:-1]


def convert_parse_tree_to_token_lst(parse_tree:list, special_tokens, var_term_to_tok_dic):
  var_term_to_tok_dic['('] = special_tokens[0]
  var_term_to_tok_dic[')'] = special_tokens[1]
  token_lst = []
  for symbol in parse_tree.split(' '):
    token_lst.append(var_term_to_tok_dic[symbol])
  return token_lst


def make_one_nontoken_abduct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, num_sents, timeout_time):
  _, _, sentss, derivss = gen_rules_grammar_MULTIPLE_sent_deriv_given_one_depth(depth=depth, num_to_gen=num_sents, num_rules=num_rules, 
                                                                                num_variables=num_variables, num_terminals=num_terminals, 
                                                                                max_rhs_len=max_rhs_len, timeout_time=timeout_time)
  assert len(sentss) == num_sents

  def make_rules_from_derivss(derivss):
    rules_set = set()
    rules = []
    for deriv in derivss:
      for r in deriv:
        if r not in rules_set:
          rules.append(r)
          rules_set.add(r)
    return rules

  rules = make_rules_from_derivss(derivss)
  return sentss, rules


def make_one_nontoken_induct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time):
  ruless, grammar, sent, deriv = gen_rules_grammar_sent_deriv_given_one_depth(depth=depth, num_rules=num_rules, 
                                                                            num_variables=num_variables, num_terminals=num_terminals, 
                                                                            max_rhs_len=max_rhs_len, timeout_time=timeout_time)
  parse_tree_string = convert_deriv_to_parse_tree_string(deriv)

  return ruless, sent, parse_tree_string


def make_one_nontoken_deduct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time):
  ruless, grammar, sent, deriv = gen_rules_grammar_sent_deriv_given_one_depth(depth=depth, num_rules=num_rules, 
                                                                            num_variables=num_variables, num_terminals=num_terminals, 
                                                                            max_rhs_len=max_rhs_len, timeout_time=timeout_time)
  return deriv, sent


def gen_rules_grammar_MULTIPLE_sent_deriv_given_one_depth(depth:int, num_to_gen, num_rules, num_variables, num_terminals, max_rhs_len, timeout_time=100000):
  '''
  make sure generated rules result in NONEMPTY sent and deriv of given depth
  '''

  while True:
    try: 
      with timeout(seconds=timeout_time):
        sentss = []
        while len(sentss) == 0:
          ruless = sample_cfg_rules(num_rules, num_variables, num_terminals, max_rhs_len)
          # shuffle rules from index 1 to end, so grammar created knows what start token is
          tmp = ruless[1:]
          random.shuffle(tmp)
          ruless[1:] = tmp

          grammar = CFG.fromstring(ruless)
          sentss, derivss = gen_multiple_sent_deriv_given_grammar_and_one_depth(grammar, depth, num_to_gen)
      break
    except TimeoutError:
      # print('timeouterror')
      continue

  return ruless, grammar, sentss, derivss


def gen_rules_grammar_sent_deriv_given_one_depth(depth:int, num_rules, num_variables, num_terminals, max_rhs_len, timeout_time=100000):
  '''
  make sure generated rules result in NONEMPTY sent and deriv of given depth
  '''
  ruless, grammar, sentss, derivss = gen_rules_grammar_MULTIPLE_sent_deriv_given_one_depth(depth, 1, num_rules, num_variables, num_terminals, max_rhs_len, timeout_time=timeout_time)
  return ruless, grammar, sentss[0], derivss[0]


def gen_sent_deriv_given_grammar_and_one_depth(grammar, depth):
  sentss, derivss = gen_multiple_sent_deriv_given_grammar_and_one_depth(grammar, depth, 1)
  if len(sentss) == 0:
    return [], []
  sent = sentss[0]
  deriv = derivss[0]

  return sent, deriv


def gen_multiple_sent_deriv_given_grammar_and_one_depth(grammar, depth, n):
  sentss, derivss = generate_specific_depth_sents_and_derivs(grammar, [grammar.start()], depth)

  if len(sentss) < n:
    return [], []

  n_sent = []
  n_deriv = []

  n_rand_idx = random.sample(range(len(sentss)), k=n)
  for rand_idx in n_rand_idx:
    n_sent.append(sentss[rand_idx])
    n_deriv.append(derivss[rand_idx])

  return n_sent, n_deriv


def gen_rules_sents_derivs_given_list_of_depths(depthss:list, num_rules, num_variables, num_terminals, max_rhs_len):
  depthss = sorted(depthss, reverse=True)
  sents_corresponding_to_depthss = []
  derivs_corresponding_to_depthss = []

  max_depth = depthss[0]

  ruless, grammar, sent, deriv = gen_rules_grammar_sent_deriv_given_one_depth(max_depth, num_rules, num_variables, num_terminals, max_rhs_len)

  sents_corresponding_to_depthss.append(sent)
  derivs_corresponding_to_depthss.append(deriv)

  for depth in depthss[1:]:
    sent, deriv = gen_sent_deriv_given_grammar_and_one_depth(grammar, depth)
    sents_corresponding_to_depthss.append(sent)
    derivs_corresponding_to_depthss.append(deriv)

  return ruless, sents_corresponding_to_depthss, derivs_corresponding_to_depthss


def convert_one_nontoken_induct_data_to_t5_ready_induct_data(rules: list, sent:list, parse_tree:str, special_tokens, vocab_tokens, num_vars, num_terms, debug=False):
  var_term_to_tok_dic = sample_var_term_to_tok_dic(vocab_tokens, num_vars, num_terms, print_dic=debug)
  rules_sent_sep_token = special_tokens[0]

  t5_rules = convert_rules_to_token_lst(rules, special_tokens[1:], var_term_to_tok_dic)
  t5_sent = convert_sent_to_token_lst(sent, var_term_to_tok_dic)
  src = t5_rules + [rules_sent_sep_token] + t5_sent

  tgt = convert_parse_tree_to_token_lst(parse_tree, special_tokens[1:], var_term_to_tok_dic)
  return src, tgt


# from cfg_utils import sample_ABDUCT_var_term_to_tok_dic
from cfg_utils import get_vars_lst
def convert_one_nontoken_abduct_data_to_t5_ready_abduct_data(sentss:list, rules:list, special_tokens, vocab_tokens, num_vars, num_terms, debug=False):
  var_term_to_tok_dic = sample_var_term_to_tok_dic(vocab_tokens, num_vars, num_terms, print_dic=debug)
  
  tgt = convert_rules_to_token_lst(rules, special_tokens, var_term_to_tok_dic)


  # SRC ===== var_tokens <SEP> sentences
  vars_lst = get_vars_lst(num_vars)
  var_toks = []
  for v in vars_lst:
    if var_term_to_tok_dic[v] in tgt:
      var_toks.append(var_term_to_tok_dic[v])

  var_toks_sents_sep_token = special_tokens[0]
  src = var_toks + [var_toks_sents_sep_token] + convert_sentss_to_token_lst(sentss, special_tokens[1:], var_term_to_tok_dic)

  return src, tgt


def convert_deriv_to_token_lst(deriv, special_tokens, var_term_to_tok_dic):
  return convert_rules_to_token_lst(deriv, special_tokens, var_term_to_tok_dic)


def convert_one_nontoken_deduct_data_to_t5_ready_deduct_data(deriv: list, sent:list, special_tokens, vocab_tokens, num_vars, num_terms, debug=False):
  var_term_to_tok_dic = sample_var_term_to_tok_dic(vocab_tokens, num_vars, num_terms, print_dic=debug)
  src = convert_deriv_to_token_lst(deriv, special_tokens, var_term_to_tok_dic)
  tgt = convert_sent_to_token_lst(sent, var_term_to_tok_dic)
  return src, tgt


import sys
import os
sys.path.append(os.getcwd())
from tasks_with_tokens import T5_TOKEN_IDS_FOR_TASKS, EXTRA_IDS

SPECIAL_TOKENS = list(EXTRA_IDS)
VOCAB_TOKENS = list(T5_TOKEN_IDS_FOR_TASKS)

def gen_one_t5_ready_deduct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time):
  deriv, sent = make_one_nontoken_deduct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time)
  return convert_one_nontoken_deduct_data_to_t5_ready_deduct_data(deriv, sent, SPECIAL_TOKENS, VOCAB_TOKENS, num_variables, num_terminals)


def gen_one_t5_ready_induct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time):
  rules, sent, parse_tree = make_one_nontoken_induct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, timeout_time)
  return convert_one_nontoken_induct_data_to_t5_ready_induct_data(rules, sent, parse_tree, SPECIAL_TOKENS, VOCAB_TOKENS, num_variables, num_terminals)


# (sentss:list, rules:list, special_tokens, vocab_tokens, num_vars, num_terms, debug=False):
def gen_one_t5_ready_abduct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, num_sents, timeout_time):
  sentss, rules = make_one_nontoken_abduct_data(num_rules, num_variables, num_terminals, max_rhs_len, depth, num_sents, timeout_time)
  return convert_one_nontoken_abduct_data_to_t5_ready_abduct_data(sentss, rules, SPECIAL_TOKENS, VOCAB_TOKENS, num_variables, num_terminals)


def get_gen_one_data_func(task):
  if task == 'induct':
    return gen_one_t5_ready_induct_data
  elif task == 'deduct':
    return gen_one_t5_ready_deduct_data
  elif task == 'abduct':
    return gen_one_t5_ready_abduct_data
  else:
    raise NotImplementedError


def generate_data(amount_to_generate, data_save_path, task, params_dic):
  gen_one_data_func = get_gen_one_data_func(task)

  data_file = open(data_save_path, mode='w')
  for i in range(amount_to_generate):
    if i%1000 == 1:
      print(f'{i} data written to {data_save_path}')
    src, tgt = gen_one_data_func(**params_dic)
    data_file.write(str(src) + '\t' + str(tgt) +'\n') 

  data_file.close()

# class SrcTooLongError(Exception):
#   pass


# class TgtTooLongError(Exception):
#   pass


import argparse
from multiprocessing import Process
import os
import os.path as osp
import sys

def str_to_bool(value):
  if isinstance(value, bool):
      return value
  if value.lower() in {'false', 'f', '0', 'no', 'n'}:
      return False
  elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
      return True
  raise ValueError(f'{value} is not a valid boolean value')


def concat_all_gen_data(data_pathss):
  concat_data_path = osp.join(data_dir, f'all_processes_{num_data_to_gen}_data_concat_{task}.txt')
  data_pathss = ' '.join(data_pathss)
  os.system(f'cat {data_pathss} > {concat_data_path}')


if __name__ == "__main__":
  # parser.add_argument('--foo', type=str_to_bool, nargs='?', const=True, default=False)
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_processes", type=int, required=True)
  parser.add_argument("--num_data_to_gen", type=int, required=True)
  parser.add_argument("--data_dir", type=str, required=True)
  parser.add_argument("--task", type=str, required=True)

  parser.add_argument("--task_params", type=str, required=True)
  parser.add_argument("--is_delete_existing_data_dir", type=str_to_bool, nargs='?', const=False, default=False)
  args = parser.parse_args()

  processes = []
  data_dir = args.data_dir
  num_processes = args.num_processes
  num_data_to_gen = args.num_data_to_gen
  task = args.task
  assert task in ['induct', 'abduct', 'deduct']
  _params = [int(a) for a in args.task_params.split(',')]

  def make_task_params_dic(params):
    if task == 'induct':
      param_names = ['num_rules', 'num_variables', 'num_terminals', 'max_rhs_len', 'depth', 'timeout_time']
    elif task == 'deduct':
      param_names = ['num_rules', 'num_variables', 'num_terminals', 'max_rhs_len', 'depth', 'timeout_time']
    elif task == 'abduct':
      # (num_rules, num_variables, num_terminals, max_rhs_len, depth, num_sents, timeout_time):
      param_names = ['num_rules', 'num_variables', 'num_terminals', 'max_rhs_len', 'depth', 'num_sents', 'timeout_time']
    else:
      raise NotImplementedError
    assert len(param_names) == len(params)
    params_dic = {k:v for k,v in zip(param_names, params)}
    return params_dic

  params_dic = make_task_params_dic(_params)
  


  print(args)
  if args.is_delete_existing_data_dir:
    os.system(f'rm -rf {data_dir}')
  else:
    print(data_dir)
    assert not os.path.isdir(data_dir)

  os.makedirs(data_dir, exist_ok=True)
  python_command_ran = 'python ' + " ".join(sys.argv)
  PYTHON_COMMAND_RAN = python_command_ran
  with open(os.path.join(data_dir, 'python_command_ran.sh'), mode='w') as f:
    f.write(python_command_ran)


  
  lst_of_per_process_amount_to_gen = []
  for i in range(num_processes):
    per_process_amount_to_gen = int(num_data_to_gen/num_processes)
    if i == (num_processes - 1):
      per_process_amount_to_gen = num_data_to_gen \
                  - (int(num_data_to_gen/num_processes) * (num_processes - 1))
    print(f'---- process {i}')
    print(per_process_amount_to_gen)
    lst_of_per_process_amount_to_gen.append(per_process_amount_to_gen)
  
  print(sum(lst_of_per_process_amount_to_gen))
  assert sum(lst_of_per_process_amount_to_gen) == num_data_to_gen

  num_processes = num_processes

  process_idss = np.random.choice(list(range(999999)), num_processes, replace=False)
  data_save_pathss = [osp.join(data_dir, f'process{process_id}_generated_{per_process_amount_to_gen}_{task}_data.txt') for process_id in process_idss]
  print(data_save_pathss)

  for i in range(num_processes):
    # per_process_amount_to_gen = int(num_data_to_gen/num_processes)
    # if i == (num_processes - 1):
    #     per_process_amount_to_gen = num_data_to_gen \
    #                 - (int(num_data_to_gen/num_processes) * (num_processes - 1))
    per_process_amount_to_gen = lst_of_per_process_amount_to_gen[i]
    data_save_path = data_save_pathss[i]
    process = Process(target=generate_data, args=(per_process_amount_to_gen, data_save_path,
                                                   task, params_dic))
    processes.append(process)
    process.start()
    

  for proc in processes:
      proc.join()


  concat_all_gen_data(data_save_pathss)

  