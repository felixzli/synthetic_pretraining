import sys
from nltk.grammar import Nonterminal, is_terminal
from cfg_utils import is_term_str, is_var_str, get_start_var
from timeout import timeout


def find_first_consecutive_terminals_from_item_list(items):
  terms = []
  for item in items:
    if isinstance(item, Nonterminal):
      return terms
    else:
      terms.append(item)
  return terms


def find_idx_first_var(items):
  for idx, item in enumerate(items):
    if isinstance(item, Nonterminal):
      return idx
  return -sys.maxsize


def cross_product_2_lst_of_lsts(lst_of_lsts_1, lst_of_lsts_2):
  result = []
  # if (lst_of_lsts_1 == [] and lst_of_lsts_2 == [[]]) or (lst_of_lsts_1 == [[]] and lst_of_lsts_2 == []):
  #   return []
  if lst_of_lsts_1 == []:
    return lst_of_lsts_2
  if lst_of_lsts_2 == []:
    return lst_of_lsts_1
  for l1 in lst_of_lsts_1:
    for l2 in lst_of_lsts_2:
      result.append(l1 + l2)
  return result


def cross_product_3_lst_of_lsts(lst_of_lsts_1, lst_of_lsts_2, lst_of_lsts_3):
  result = []
  for l1 in lst_of_lsts_1:
    for l2 in lst_of_lsts_2:
      for l3 in lst_of_lsts_3:
        result.append(l1 + l2 + l3)
  return result


def generate_specific_depth_sents_and_derivs(grammar, items, depth):
  '''
  return: list of sents, list of derivs

  deriv is list of str(prod)
  sent is a list of terms
  '''
  if all([is_terminal(i) for i in items]):
    if depth == 0:
      return [items], [[]]
    else:
      return [], []
  elif depth == 0:
    return [], []
  else:
    sents = []
    derivs = []
    get_idx_next_var = lambda idx_processed_so_far: (idx_processed_so_far + 1) + find_idx_first_var(items[(idx_processed_so_far+1):])
    get_terms_before_var = lambda idx_processed_so_far, idx_next_var: items[(idx_processed_so_far+1):(idx_next_var)]

    idx_processed_so_far = -1
    idx_next_var = get_idx_next_var(idx_processed_so_far)

    while idx_next_var >= 0:
      var = items[idx_next_var]
      terms_before_var = get_terms_before_var(idx_processed_so_far, idx_next_var)
      sents = cross_product_2_lst_of_lsts(sents, [terms_before_var])

      # print(f'sents {sents}')
      # sents_before_expand_var = sents.copy()
      # derivs_before_expand_var = derivs.copy()

      var_all_sents = []
      var_all_derivs = []
      # print(grammar.productions(var))
      for prod in grammar.productions(var):
        var_sents, var_derivs = generate_specific_depth_sents_and_derivs(grammar, list(prod.rhs()), depth-1)
        if var_sents == []:
          continue
        str_prod = str(prod)
        var_derivs = cross_product_2_lst_of_lsts([[str_prod]], var_derivs)
        var_all_sents += var_sents
        var_all_derivs += var_derivs
      

      if var_all_sents == []:
        return [], []

      sents = cross_product_2_lst_of_lsts(sents, var_all_sents)
      derivs = cross_product_2_lst_of_lsts(derivs, var_all_derivs)
      
      idx_processed_so_far = idx_next_var
      idx_next_var = get_idx_next_var(idx_processed_so_far)

    if idx_processed_so_far != len(items) - 1:
      sents = cross_product_2_lst_of_lsts(sents, [items[idx_processed_so_far+1:len(items)]])

    return sents, derivs


class ParseTreeNode:
  def __init__(self, value, children=[]):
    '''
    children: list of ParseTreeNode
    '''
    self.set_value(value)
    self.set_children(children)
    assert type(self.children) == list
 
  def set_children(self, children):
    self.children = children

  def set_value(self, value):
    if value == '':
      value = 'empty_string'
    self.value = value

  def get_value(self):
    return self.value

  def is_var(self):
    return is_var_str(self.value)

  def get_children(self):
    return self.children

  def add_children(self, children_to_add):
    self.children = self.children + children_to_add

  def get_children_values(self):
    return [c.get_value() for c in self.get_children()]

  def is_leaf(self):
    return len(self.get_children()) == 0

  def __str__(self):
    result = f'value: {self.get_value()} ||| children:{self.get_children_values()}'
    return result


def get_rhs_nodes_from_rule(rule):
  _, rhs = rule.split(' -> ')
  rhs = rhs.split(' ')
  rhs_nodes = []

  for v in rhs:
    if is_term_str(v):
      rhs_nodes.append(ParseTreeNode(eval(v)))
    elif is_var_str(v):
      rhs_nodes.append(ParseTreeNode(v))
    else:
      raise NotImplementedError

  rhs_var_nodes = []
  for node in rhs_nodes:
    if node.is_var():
      rhs_var_nodes.append(node)
  return rhs_nodes, rhs_var_nodes


def get_lhs_var_from_rule(rule:str):
  return rule.split(' -> ')[0]


def get_index_of_node_with_val_in_lst_nodes(val:str, lst:list):
  for i, node in enumerate(lst):
    if node.get_value() == val:
      return i 
  raise RuntimeError(f'node with val {val} was not found in list')


def convert_deriv_to_parse_tree(deriv):
  start_var_node = ParseTreeNode(get_start_var())
  nodess_to_set_children = [start_var_node]
  
  for rule in deriv:
    lhs_var = get_lhs_var_from_rule(rule)

    node = nodess_to_set_children[0]
    assert node.get_value() == lhs_var

    rhs_nodes, rhs_var_nodes = get_rhs_nodes_from_rule(rule)
    assert len(node.get_children()) == 0
    node.set_children(rhs_nodes)

    nodess_to_set_children = rhs_var_nodes + nodess_to_set_children[1:]

  assert len(nodess_to_set_children) == 0
  return start_var_node


def _convert_tree_to_string(node):
  if node.is_leaf():
    return f' {node.get_value()} '
  else:
    tree_string = f' {node.get_value()} '

    for c in node.get_children():
      tree_string += '('
      tree_string += _convert_tree_to_string(c)
      tree_string += ') '

    return tree_string


def convert_tree_to_string(node):
  return _convert_tree_to_string(node)[1:-1]



def convert_deriv_to_parse_tree_string(deriv):
  parse_tree = convert_deriv_to_parse_tree(deriv)
  return convert_tree_to_string(parse_tree)


def get_leaves_in_order(node):
  if node.is_leaf():
    return [node.get_value()]
  else:
    leaves = []
    for c in node.get_children():
      leaves.extend(get_leaves_in_order(c))
    return leaves


if __name__ == '__main__':
  # print(cross_product_2_lst_of_lsts([[1,2,3],[4,5,6]],[[3,3],[7,7]]))
  # print(cross_product_2_lst_of_lsts([[]],[[1,2,3,4]]))
  # print(cross_product_2_lst_of_lsts([[]], [[]]))


  # parse_tree = convert_deriv_to_parse_tree(['V0 -> '])
  # print(parse_tree)
  # parse_tree = convert_deriv_to_parse_tree(["V0 -> V1 't0'",
  #                                           'V1 -> V3',
  #                                           "V3 -> V1 't0' V0",
  #                                           'V1 -> V3',
  #                                           "V3 -> 't2'",
  #                                           "V0 -> V1 't0'",
  #                                           "V1 -> 't4'"])

  # parse_tree = convert_deriv_to_parse_tree(["V0 -> V1 't0'",
  #                                           'V1 -> V3',
  #                                           "V3 -> V1 't0' V0",
  #                                           'V1 -> V3',
  #                                           "V3 -> 't2'",
  #                                           "V0 -> V1 't0'",
  #                                           'V1 -> '])


  deriv = ["V0 -> 't1' 't3' V2",
   'V2 -> V0 V1 V1',
   "V0 -> 't1' 't3' V2",
   'V2 -> V0 V1 V1',
   "V0 -> 't3'",
   "V1 -> 't1'",
   "V1 -> 't1'",
   "V1 -> V1 't1' V4",
   "V1 -> V1 't1' V4",
   "V1 -> 't1'",
   "V4 -> 't4'",
   'V4 -> V0 V0',
   "V0 -> 't3'",
   "V0 -> 't3'",
   "V1 -> V1 't1' V4",
   "V1 -> V1 't1' V4",
   "V1 -> 't1'",
   "V4 -> 't4'",
   'V4 -> V0 V0',
   "V0 -> 't3'",
   "V0 -> 't3'"]

  parse_tree = convert_deriv_to_parse_tree(deriv)

  print(convert_tree_to_string(parse_tree))
  print(get_leaves_in_order(parse_tree))

  print(convert_deriv_to_parse_tree_string(deriv))


  
  
