from numpy import random
from nltk import CFG
import random
import numpy as np
from nltk.parse import RecursiveDescentParser
from nltk.parse import ShiftReduceParser


def _random_partition(n, k):
  """
  Return random partition n with k parts
	Example: (7,3) -> [2,4,1]
	"""
  result = np.random.multinomial(n, np.ones(k)/k, size=1)[0]
  result = [int(i) for i in result]
  return result


def _random_partition_no_zeros(n, k):
  """
  Return random partition n with k nonzero parts
	Example: (7,3) -> [2,4,1]
	"""
  result = np.random.multinomial(n - k, np.ones(k)/k, size=1)[0] + np.ones(k)
  result = [int(i) for i in result]
  return result


def get_vars_lst(num_variables):
  return [f"V{i}" for i in range(num_variables)]


def get_terms_lst(num_terminals):
  return [f"'t{i}'" for i in range(num_terminals)]


def sample_var_term_toks(vocab_tokens, num_vars, num_terms):
  # assert len(vocab_tokens) == (num_vars + num_terms)
  vars_terms = np.random.choice(vocab_tokens, size=num_vars+num_terms, replace=False)
  return vars_terms[:num_vars], vars_terms[num_vars:num_vars+num_terms]


def sample_var_term_to_tok_dic(vocab_tokens, num_vars, num_terms, print_dic=False):
  var_tokens, term_tokens = sample_var_term_toks(vocab_tokens, num_vars, num_terms)
  assert len(var_tokens) == num_vars
  assert len(term_tokens) == num_terms
  dic = {}
  vars_lst = get_vars_lst(num_vars)
  terms_lst = get_terms_lst(num_terms)

  for v, vt in zip(vars_lst, var_tokens):
    dic[v] = vt
  for t, tt in zip(terms_lst, term_tokens):
    dic[eval(t)] = tt

  if print_dic:
    print(f'dic from sample_var_term_to_tok_dic: {dic}')
  return dic


def sample_ABDUCT_var_term_to_tok_dic(vocab_tokens, num_vars, num_terms, print_dic=False):
  var_tokens = vocab_tokens[:num_vars]
  _, term_tokens = sample_var_term_toks(vocab_tokens[num_vars:], num_vars, num_terms)

  assert len(var_tokens) == num_vars
  assert len(term_tokens) == num_terms

  dic = {}
  vars_lst = get_vars_lst(num_vars)
  terms_lst = get_terms_lst(num_terms)

  for v, vt in zip(vars_lst, var_tokens):
    dic[v] = vt
  for t, tt in zip(terms_lst, term_tokens):
    dic[eval(t)] = tt

  if print_dic:
    print(f'dic from sample_var_term_to_tok_dic: {dic}')
  return dic




def is_term_str(s):
  return 't' in s


def is_var_str(s):
  return 'V' in s


ARROW_SYMBOL = '->'
def sample_cfg_rules(num_rules, num_variables, num_terminals, max_rhs_len):
  '''
  details:
  don't allow identity rule
  dont allow unreachable rules. (LHS vars are only either start_var or sampled from vars that have appeared on RHS)
  don't allow var to single var rule
  add rules mapping each var to a term
  don't allow duplicate rules

  '''
  # sampling described here: https://docs.google.com/document/d/1cdF1Qjti8CzXeujsGU_42zScZusn18s2aUde1MAVDDs/edit#
  # assert num_terminals >= num_variables
  assert max_rhs_len >= 2
  rules_set = set()
  rules_lst = []

  vars = get_vars_lst(num_variables)
  start_var = get_start_var()
  vars_excluding_start_var = vars[1:]
  terms = get_terms_lst(num_terminals)

  # vars_and_terms = vars + terms
  
  vars_sample_with_replace = lambda n: random.choices(vars, k=n) 
  terms_sample_with_replace = lambda n: random.choices(terms, k=n) 

  used_vars = set()
  used_vars.add(start_var)
  for _ in range(num_rules):
    rhs_len = random.randint(2, max_rhs_len)    
    lhs = random.choices(list(used_vars), k=1)[0]

    num_rhs_vars, num_rhs_terms = _random_partition(rhs_len, 2)
    rhs_vars = vars_sample_with_replace(num_rhs_vars)

    # prevent identity rule
    if rhs_len == 1 and num_rhs_vars == 1:
      while rhs_vars[0] == lhs:
        rhs_vars = vars_sample_with_replace(num_rhs_vars)

    rhs_terms = terms_sample_with_replace(num_rhs_terms)
    rhs = rhs_vars + rhs_terms
      
    for v in rhs_vars:
        used_vars.add(v)

    assert type(rhs) == list

    random.shuffle(rhs)
    rhs = " ".join(rhs)
    rule = f'{lhs} {ARROW_SYMBOL} {rhs}'

    # don't allow duplicate rules
    if rule not in rules_set:
      rules_set.add(rule)
      rules_lst.append(rule)

  # used_vars_len_terms = terms[:len(used_vars)]
  # random.shuffle(used_vars_len_terms)
  # for var, term in zip(used_vars, used_vars_len_terms):
  #   rules_lst.append(f"{var} {ARROW_SYMBOL} {term}")


  for var in used_vars:
    rand_i = random.randint(0, len(terms) - 1)
    rand_term = terms[rand_i]
    rule = f"{var} {ARROW_SYMBOL} {rand_term}"
    if rule not in rules_set:
      rules_lst.append(rule)

  return rules_lst


def get_start_var():
  return 'V0'


if __name__ == '__main__':
  for _ in range(30):
    print('-----')
    # print(sample_cfg_rules(num_rules, num_variables, num_terminals, max_rhs_len))
    [print(r) for r in sample_cfg_rules(8, 10, 3, 3)]


if __name__=='__main__':
  print(sample_var_term_to_tok_dic(list(range(30)), 2, 4))
  print(sample_ABDUCT_var_term_to_tok_dic(list(range(30)), 10, 4))