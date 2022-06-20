import unittest
from generate_cfg_task_data import *
from nltk import CFG
from timeout import timeout, TimeoutError
import time


class TestTimeToGenerate100(unittest.TestCase):
  def test_gen_rules_grammar_sent_deriv_given_one_depth(self):
    # depth:int, num_rules, num_variables, num_terminals, max_rhs_len
    # return
    for i in range(100):
      print(i)
      try:
        with timeout(seconds=5):
          r, g, s, d = gen_rules_grammar_sent_deriv_given_one_depth(4, 7, 5, 5, 4)
          print(s)
      except TimeoutError as e:
        print('timeout error')
        continue

  # def test_freeze_causing_rule(self):
  #   freeze_causing_rules = ["V0 -> V0 V0 't1'", 'V0 -> V0 V2', "V0 -> 't4' V1", "V2 -> 't3' 't3'", 'V2 -> V0 V2', "V2 -> 't2' 't1'", "V1 -> 't3' V0", "V0 -> 't2'", "V2 -> 't0'", "V1 -> 't1'"]
  #   grammar = CFG.fromstring(freeze_causing_rules)
  #   gen_sent_deriv_given_grammar_and_one_depth(grammar, 5)


  # def test_isupper(self):
  #   self.assertTrue('FOO'.isupper())
  #   self.assertFalse('Foo'.isupper())



class PrintNonTokenData(unittest.TestCase):
  def test_print_make_one_nontoken_induct_data(self):
    for _ in range(2):
      print('-'*10)
      ruless, sent, parse_tree_string = make_one_nontoken_induct_data(7,5,5,3,4, 1)
      print(ruless)
      print(sent)
      print(parse_tree_string)

  # python cfg_tasks/test_generate_cfg_task_data.py PrintNonTokenData.test_print_make_one_nontoken_abduct_data 
  def test_print_make_one_nontoken_abduct_data(self):
    for _ in range(10):
      print('-'*10)
      sentss, rules = make_one_nontoken_abduct_data(4,3,3,3,3,20,5)
      [print(sent) for sent in sentss]
      print(rules)


  def test_make_one_nontoken_deduct_data(self):
    for _ in range(2):
      deriv, sent = make_one_nontoken_deduct_data(7,5,5,3,4,1)
      print('-'*10)
      print(deriv)
      print(sent)


class PrintGenThenConvertNonTokenData(unittest.TestCase):
  special_tokens = [999,888,777]
  vocab_tokens = list(range(10))
  def test_induct(self):
    print('\n'*2)
    print('-'*10)
    ruless, sent, parse_tree_string = make_one_nontoken_induct_data(7,2,2,3,3,1)
    print(ruless)
    print(sent)
    print(parse_tree_string)

    src, tgt = convert_one_nontoken_induct_data_to_t5_ready_induct_data(ruless, sent, parse_tree_string, self.special_tokens, self.vocab_tokens, 2, 2, debug=True)
    print('-'*10)
    print(src)
    print(tgt)

  def test_abduct(self):
    print('\n'*2)
    sentss, rules = make_one_nontoken_abduct_data(3,2,2,3,depth=3, num_sents=3, timeout_time=1)
    print('-'*10)
    [print(sent) for sent in sentss]
    print(rules)

    src, tgt = convert_one_nontoken_abduct_data_to_t5_ready_abduct_data(sentss, rules, self.special_tokens, self.vocab_tokens, 2, 2, debug=True)
    print('-'*10)
    print(src)
    print(tgt)


  def test_deduct(self):
    print('\n'*2)
    deriv, sent = make_one_nontoken_deduct_data(3,2,2,3,3,1)
    print('-'*10)
    print(deriv)
    print(sent)

    src, tgt = convert_one_nontoken_deduct_data_to_t5_ready_deduct_data(deriv, sent, self.special_tokens, self.vocab_tokens, 2, 2, debug=True)
    print('-'*10)
    print(src)
    print(tgt)

class TestConvertNontokenToTokenLstFuncs(unittest.TestCase):
  def test_convert_rules_to_token_lst(self):    
    rules = ["V0 -> V1 V4 't3'", "V4 -> 't4' V4", "V3 -> 't3'"]
    token_lst = convert_rules_to_token_lst(rules, [999,888], 
                      {'V0':0, 'V1': 1, 'V2':2, 'V3': 3, 'V4': 4, 
                      "t0": 10, "t1": 11, "t2": 12, "t3": 13, "t4":14}) 

    expected_token_lst = [0, 999, 1, 4, 13, 888, 4, 999, 14, 4, 888, 3, 999, 13]
    print('')
    print(token_lst)
    print(expected_token_lst)
    self.assertEqual(token_lst, expected_token_lst)

  def test_convert_sentss_to_token_lst(self):    
    sentss = [['t1', 't1', 't0'],
              ['t1', 't1'],
              ['t1', 't4', 't4', 't3']]
    token_lst = convert_sentss_to_token_lst(sentss, [999,888], 
                      {'V0':0, 'V1': 1, 'V2':2, 'V3': 3, 'V4': 4, 
                      "t0": 10, "t1": 11, "t2": 12, "t3": 13, "t4":14}) 

    expected_token_lst = [11, 11, 10, 999, 11, 11, 999, 11, 14, 14, 13]
    print('')
    print(token_lst)
    print(expected_token_lst)
    self.assertEqual(token_lst, expected_token_lst)


  def test_convert_sent_to_token_lst(self):    
    sent = ['t1', 't1', 't0']
    token_lst = convert_sent_to_token_lst(sent, 
                      {'V0':0, 'V1': 1, 'V2':2, 'V3': 3, 'V4': 4, 
                      "t0": 10, "t1": 11, "t2": 12, "t3": 13, "t4":14}) 

    expected_token_lst = [11, 11, 10]
    print('')
    print(token_lst)
    print(expected_token_lst)
    self.assertEqual(token_lst, expected_token_lst)


  def test_convert_parse_tree_to_token_lst(self):    
    _, _, parse_tree =  make_one_nontoken_induct_data(7,4,4,3,2,1)
    # print(parse_tree)
    assert parse_tree[0] != ' '
    assert parse_tree[1] != ' '

    parse_tree = "V0 ( t1 ) ( V0 ( t2 ) )"
    print(parse_tree)
    token_lst = convert_parse_tree_to_token_lst(parse_tree, [999,888], 
                      {'V0':0, 'V1': 1, 'V2':2, 'V3': 3, 'V4': 4, 
                      "t0": 10, "t1": 11, "t2": 12, "t3": 13, "t4":14}) 

    expected_token_lst = [0, 999, 11, 888, 999, 0, 999, 12, 888, 888]
    print('')
    print(token_lst)
    print(expected_token_lst)
    self.assertEqual(token_lst, expected_token_lst)


if __name__ == '__main__':
    unittest.main()

    # python cfg_tasks/test_generate_cfg_task_data.py Test.test_print_make_one_nontoken_induct_data 
    # python cfg_tasks/test_generate_cfg_task_data.py Test.test_gen_rules_grammar_sent_deriv_given_one_depth_no_timeout
    # python cfg_tasks/test_generate_cfg_task_data.py TestTimeToGenerate100
    # python cfg_tasks/test_generate_cfg_task_data.py PrintNonTokenData