import abc
import os
import os.path as osp
import numpy as np
import random
import collections
from argparse import Namespace

from collections import namedtuple

from numpy.random.mtrand import rand

TASK_REGISTRY = {}
Task = namedtuple("Task", ['name', 'description', 'sampling_fn', 'fn', 'unary'])
SAMPLING_FUNCTIONS = []
UPPER_AND_LOWER_LETTERS = set([chr(i) for i in range(ord('a'), ord('z')+1)] \
                              + [chr(i) for i in range(ord('A'), ord('Z')+1)])
WORD_DIVIDER = ','

### util functions ###
def _str_to_data(string):
  # string = str(string)
  if len(string) == 0:
    return ''
  # return ' '.join([str(ord(ch)) for ch in str(string)])
  return ' '.join([c for c in string])


def _data_to_str(ascii):
  if len(ascii) == 0:
    return ''
  # return ''.join([chr(int(x)) for x in ascii.split(' ')])
  return ''.join(ascii.split(' '))


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

def _get_char_set(low=ord('A'), high=ord('Z')):
  return set([str(i) for i in range(low, high+1)])


### sampling function ###

def _generate_random_str_list_given_length(length, char_set):
  return [str(i) for i in random.choices(tuple(char_set), k=length)]


def _generate_random_str_given_length(length, char_set):
  return " ".join(_generate_random_str_list_given_length(length, char_set))


def generate_random_str_given_range(length_range, char_set):
  seq_len = random.randint(*length_range)
  return _generate_random_str_given_length(seq_len, char_set)
SAMPLING_FUNCTIONS.append(generate_random_str_given_range)


def generate_count_data(length_range, char_set):
  assert length_range[0] > 2
  # subtract by 2 so final length of {seq <sep> query_char} is within length range
  seq_char_count = random.randint(length_range[0]-2, length_range[1]-2)
  if seq_char_count == 0:
    return ''
  
  char_set = char_set.copy()
  query = random.choice(tuple(char_set))
  query_char_count = random.randint(0, seq_char_count)
  not_query_char_count = seq_char_count - query_char_count

  char_set.discard(query)
  data_list = [query]*query_char_count \
              + _generate_random_str_list_given_length(not_query_char_count, char_set)
  random.shuffle(data_list)
  return ' '.join(data_list), query
SAMPLING_FUNCTIONS.append(generate_count_data)


def _generate_deduplicate_data(length_range, char_set, is_shuffle, list_chars_to_dup = None, fixed_data_character_count=None):
  if fixed_data_character_count is None:
    data_char_count = random.randint(*length_range)
  else: 
    data_char_count = fixed_data_character_count
  if data_char_count == 0:
    return ''
    
  char_set = char_set.copy()
  if list_chars_to_dup is None:
    num_parts = random.randint(1, data_char_count)
    partition = _random_partition_no_zeros(data_char_count, num_parts)
    num_parts_length_seq = _generate_random_str_list_given_length(num_parts, char_set)
  else:
    num_parts = len(list_chars_to_dup)
    partition = _random_partition_no_zeros(data_char_count, num_parts)
    num_parts_length_seq = random.choices(list_chars_to_dup, k=num_parts)
  data = []

  for i, part in enumerate(partition):
    data += [num_parts_length_seq[i]] * part

  if is_shuffle:
    random.shuffle(data)
  assert 0 not in partition
  assert sum(partition) == data_char_count
  assert len(data) == data_char_count

  data = ' '.join(data)
  return data


def generate_deduplicate_data(length_range, char_set):
  return _generate_deduplicate_data(length_range, char_set, False)
SAMPLING_FUNCTIONS.append(generate_deduplicate_data)


def generate_set_data(length_range, char_set):
  return _generate_deduplicate_data(length_range, char_set, True)
SAMPLING_FUNCTIONS.append(generate_set_data)


def generate_longest_word_data(length_range, char_set):
  assert length_range[0] >= 1
  # min word length = 1
  # num_spaces = num_words - 1 
  # spaces are currently represented by commas
  # ----> max_num_words + (max_num_words - 1) = length_range[1]
  max_num_words = int(length_range[1]/2)
  num_words = random.randint(1, max_num_words)
  num_space = num_words - 1

  max_num_not_space = length_range[1] - num_space
  min_num_not_space = max(length_range[0] - num_space, num_words)
  num_not_space = random.randint(min_num_not_space, max_num_not_space)

  space = WORD_DIVIDER
  char_set_no_space = char_set.copy()
  char_set_no_space.discard(space)

  data = []
  word_lengths = _random_partition_no_zeros(num_not_space, num_words)
  for i, wl in enumerate(word_lengths):
    data += _generate_random_str_list_given_length(wl, char_set_no_space)
    if i < num_space:
      data += [space]
  return ' '.join(data)
SAMPLING_FUNCTIONS.append(generate_longest_word_data)


def generate_at_least_one_query_in_seq_data(length_range, char_set, query_seq_max_proportion):
  # max_total_chars = length_range[1] - 1 because max_total_chars doesn't count the <sep> token
  max_total_chars = length_range[1] - 1 
  # max_total_chars = seq chars + query chars 
  # max_q/qs_max_prop + max_q = max_total_chars
  # ---> max_q = max_total_chars / (1 + 1/qs_max_prop)
  max_query_length = int(max_total_chars / (1 + 1/query_seq_max_proportion))
  query_char_count = random.randint(1, max_query_length)
  seq_char_count = random.randint(query_char_count / query_seq_max_proportion, max_total_chars - query_char_count)
  char_set = char_set.copy()
  query = _generate_random_str_list_given_length(query_char_count, char_set)
  num_query_occurences = random.randint(1, int(seq_char_count / query_char_count))
  not_query_char_count = seq_char_count - num_query_occurences * query_char_count
  assert not_query_char_count >= 0
  seq = []
  partition = _random_partition(not_query_char_count, num_query_occurences+1)
  assert sum(partition) == not_query_char_count
  for i, part in enumerate(partition):
    seq += _generate_random_str_list_given_length(part, char_set)
    if i < num_query_occurences:
      seq += query
  assert len(seq) == seq_char_count
  return ' '.join(seq), ' '.join(query)


def generate_filter_data(length_range, char_set, query_seq_max_proportion=0.5):
  if random.uniform(0, 1) > 0.25:
    return generate_at_least_one_query_in_seq_data(length_range, char_set, query_seq_max_proportion )
  else:
    return generate_no_query_in_seq_data(length_range, char_set)


def generate_delete_data(length_range, char_set, query_seq_max_proportion=0.5):
  if random.uniform(0, 1) > 0.25:
    return generate_at_least_one_query_in_seq_data(length_range, char_set, query_seq_max_proportion )
  else:
    return generate_no_query_in_seq_data(length_range, char_set)


def generate_get_index_data(length_range, char_set, query_seq_max_proportion=0.5):
  if random.uniform(0, 1) > 0.25:
    return generate_at_least_one_query_in_seq_data(length_range, char_set, query_seq_max_proportion )
  else:
    return generate_no_query_in_seq_data(length_range, char_set)
SAMPLING_FUNCTIONS.extend([generate_delete_data, generate_get_index_data])


def generate_no_query_in_seq_data(length_range, char_set):
  # 1% chance sample data with query length > seq length
  propor_query_len_gt_seq_len = 0.01
  if random.uniform(0, 1) < propor_query_len_gt_seq_len:
    query_char_count = random.randint(length_range[0] + 1, length_range[1])
    query = generate_random_str_given_range((query_char_count, query_char_count), char_set)
    seq_char_count = random.randint(0, query_char_count - 1)
    seq = generate_random_str_given_range((seq_char_count, seq_char_count), char_set)
    return seq, query
  
  # remaining chance sample data with query length <= seq length
  seq_char_count = random.randint(*length_range)
  query_char_count = random.randint(length_range[0], seq_char_count)
  query = generate_random_str_given_range((query_char_count, query_char_count), char_set)
  seq = generate_random_str_given_range((seq_char_count, seq_char_count), char_set)
  while _data_to_str(query) in _data_to_str(seq):
    query = generate_random_str_given_range((query_char_count, query_char_count), char_set)
    seq = generate_random_str_given_range((seq_char_count, seq_char_count), char_set)
  return seq, query


def generate_search_data(length_range, char_set, query_seq_max_proportion=0.5):
  if random.uniform(0, 1) > 0.5:
    return generate_at_least_one_query_in_seq_data(length_range, char_set, query_seq_max_proportion)
  else:
    return generate_no_query_in_seq_data(length_range, char_set)
SAMPLING_FUNCTIONS.append(generate_search_data)
  

def generate_sort_data(length_range, char_set):
    # - 1 because of <sep> token
    max_total_chars = length_range[1] - 1 
    # max_total_chars = seq chars + order chars 
    order_char_count = random.randint(int(length_range[0]/2), int(max_total_chars/2))
    order = random.choices(list(char_set), k=order_char_count)
    seq_char_count = random.randint(order_char_count, max_total_chars - order_char_count)
    seq = _generate_deduplicate_data(None, char_set, True, order, seq_char_count)
    return seq, ' '.join(order)
SAMPLING_FUNCTIONS.append(generate_sort_data)


def _generate_replace_query_dic_given_str_list(seq, char_set, num_replace):
    keys = random.sample(list(set(seq)), k=num_replace)
    values = _generate_random_str_list_given_length(len(keys), char_set)
    query = []
    for k, v in zip(keys, values):
      query.append(k)
      query.append(v)
    return ' '.join(query)


def generate_replace_data(length_range, char_set):
  # current data sampling: replaced key will for sure exist in seq. so need min_length of total input sequence > 3
  # min length possible input: k <sep> k v
  assert length_range[0] > 3  
  # - 3 because of <sep> key val
  min_seq_length = length_range[0] - 3
  max_seq_length = length_range[1] - 3

  seq_char_count = random.randint(min_seq_length, max_seq_length)
  seq = _generate_random_str_list_given_length(seq_char_count, char_set)
  query = _generate_replace_query_dic_given_str_list(seq, char_set, 1)
  return ' '.join(seq), query
SAMPLING_FUNCTIONS.append(generate_replace_data)


def generate_replace_many_data(length_range, char_set):
  char_set = list(char_set)
  # - 1 because of <sep> token
  max_total_chars = length_range[1] - 1 
  min_total_chars = length_range[0] - 1 

  max_num_replace = int(max_total_chars / 3)
  min_num_replace = 2
  num_replace = random.randint(min_num_replace, max_num_replace)

  seq_char_count = random.randint(max(num_replace, min_total_chars-num_replace*2), max_total_chars - num_replace*2)

  min_unique_seq_chars = random.sample(char_set, k=num_replace)
  seq = min_unique_seq_chars + random.choices(char_set, k=seq_char_count-num_replace)
  random.shuffle(seq)
  query = _generate_replace_query_dic_given_str_list(seq, char_set, num_replace)
  return ' '.join(seq), query
SAMPLING_FUNCTIONS.append(generate_replace_many_data)


def generate_two_seq_data(length_range, char_set):
  assert length_range[0] >= 3
  # - 1 to account for sep char
  max_total_chars = length_range[1] - 1 
  min_total_chars = length_range[0] - 1
  total_chars = random.randint(min_total_chars, max_total_chars)
  # seq1_char_count = random.randint(1, total_chars - 1)
  seq1_char_count = random.randint(0, total_chars)
  seq2_char_count = total_chars - seq1_char_count
  seq1 = _generate_random_str_given_length(seq1_char_count, char_set)
  seq2 = _generate_random_str_given_length(seq2_char_count, char_set)
  return seq1, seq2
SAMPLING_FUNCTIONS.append(generate_two_seq_data)



BASIC_TASK_SAMPLING_ARGS = ['length_range', 'char_set']
LIME_SAMPLING_ARGS = ['length_range', 'char_set', 'is_original_vocab_division',
                      'ida_num_chars_range', 'ida_pattern_length_range', 
                      'num_upper_case_letters', 'num_lower_case_letters', 'num_math_symbols']
SUMMARIZATION_SAMPLING_ARGS = ['length_range', 
                              'char_set', 'vocab_size', 'num_chars_per_word', 
                              'mean_numsents', 'mean_sentlen']
MBPP_SAMPLING_ARGS = ['length_range', 'char_set']
### util function for data generation ###
def generate_example(task, is_natural_language, **sampling_args):
  if 'lime' in task.name:
    assert all([arg in sampling_args for arg in LIME_SAMPLING_ARGS])
    sampled_input, output_str = task.sampling_fn(mode=task.name.split('_')[-1], 
                      **sampling_args)
  elif 'summ' in task.name:
    assert all([arg in sampling_args for arg in SUMMARIZATION_SAMPLING_ARGS])
    sampled_input, output_str = task.sampling_fn(mode=task.name.split('_')[-1], **sampling_args)
  elif task.name in MBPP_TASKS:
    assert all([arg in sampling_args for arg in MBPP_SAMPLING_ARGS])
    sampled_input, output_str = gen_mbpp_data(task, **sampling_args)
  else: # the first binary/unary tasks in the google doc
    assert all([arg in sampling_args for arg in BASIC_TASK_SAMPLING_ARGS])
    sampled_input = og_sampled_input = task.sampling_fn(**sampling_args)
    if not task.unary:
      sampled_input = " <sep> ".join(sampled_input)
    input_len = len(sampled_input.split(' '))
    while input_len < sampling_args['length_range'][0] or input_len > sampling_args['length_range'][1]:
      sampled_input = og_sampled_input = task.sampling_fn(**sampling_args)
      if not task.unary:
        sampled_input = " <sep> ".join(sampled_input)
      input_len = len(sampled_input.split(' '))
    output = task.fn(og_sampled_input)
    output_str = str(output)

  if is_natural_language:
    description = task.description
  else:
    description = task.name
  input_str = "{} {}".format(description, sampled_input)
  # delete trailing space
  if input_str[-1] == ' ': 
    input_str = input_str[:-1]
  # replace double and triple space with single space
  input_str = input_str.replace('   ', ' ').replace('  ', ' ')
  output_str = output_str.replace('   ', ' ').replace('  ', ' ')
  return input_str, output_str


def generate_batch_examples(task, num_examples, file_dir="./ulime_datasets/", is_natural_language=False, task_config_str="", **sampling_args):
  if not os.path.exists(osp.join(file_dir, task.name)):
    os.makedirs(osp.join(file_dir, task.name), exist_ok=True)
  file_name = "{}_{}M.txt".format(
      task_config_str, num_examples // 1_000_000)
  data_path = osp.join(file_dir, task.name, file_name)
  with open(data_path, "w") as f:
    for count in range(num_examples):
      example = generate_example(task, is_natural_language, **sampling_args)
      f.write("\t".join(example)+"\n")
      if count % 100_000 == 0:
        print(f'generated {task.name} {count}') 
  return data_path

### Basic unary Tasks ###

def copy(s):
  return s


def reverse(s):
  """
  Return the reverse of a string.
	Example: aaacddc -> cddcaaa
	"""

  return ' '.join(s.split(' ')[::-1])


def set_task(s):
  """
	Example: aabbbcbbbb -> abc
	"""
  char_set = set()
  new_s = ""
  for i in s:
    if i in char_set or i == ' ':
      continue
    else:
      char_set.update(i)
      new_s += i + ' '
  return new_s[:-1]


def duplicate(s):
  """
	Example: abdjw -> aabbddjjww
	"""
  s = s.split(' ')
  result = []
  for ascii in s:
    result.append(ascii)
    result.append(ascii)
  return ' '.join(result)


def deduplicate(input):
  """
	Example: aabbbcbbbb -> abcb
	"""
  input = input.split(' ')
  dedup_input = []
  last = ""
  for i in input:
    if i == last:
      continue
    else:
      last = i
      dedup_input.append(i)
  return ' '.join(dedup_input)


def first_char(s):
  """
  Return the first char in a string.
	Example: aaacddc -> a
	"""
  return s.split(' ')[0]


def last_char(s):
  """
  Return the last char in a string.
	Example: aaacddc -> c
	"""
  return s.split(' ')[-1]


def length(s):
  """
  Return the length of a string.
	Example: cjvnac -> 6
	"""
  return _str_to_data(str(len(s.split(' '))))


def longest_word(s):
  """
  Return the longest word in a string.
	Example: cjvnac asd aa -> cjvnac
	"""
  space = WORD_DIVIDER
  curr_word_length = 0
  max_length_word_list = []
  curr_word_list = []
  s = s.split(' ')
  for char in s:
    if char != space:
      curr_word_list.append(char)
      curr_word_length += 1
      if curr_word_length > len(max_length_word_list):
        max_length_word_list = curr_word_list
    else:
      curr_word_list = []
      curr_word_length = 0
  return ' '.join(max_length_word_list)


TASK_REGISTRY['copy'] = Task(
  name='copy',
  description='Copy the following string: ',
  sampling_fn=generate_random_str_given_range,
  fn=copy,
  unary=True)

TASK_REGISTRY['reverse'] = Task(
  name='reverse',
  description='Reverse the following string: ',
  sampling_fn=generate_random_str_given_range,
  fn=reverse,
  unary=True)

TASK_REGISTRY['set'] = Task(
  name='set',
  description='Extract the set of chars used in the following string: ',
  sampling_fn=generate_set_data,
  fn=set_task,
  unary=True)

TASK_REGISTRY['first_char'] = Task(
  name='first_char',
  description='Extract the first char used in the following string: ',
  sampling_fn=generate_random_str_given_range,
  fn=first_char,
  unary=True)

TASK_REGISTRY['last_char'] = Task(
  name='last_char',
  description='Extract the last char used in the following string: ',
  sampling_fn=generate_random_str_given_range,
  fn=last_char,
  unary=True)

TASK_REGISTRY['deduplicate'] = Task(
  name='deduplicate',
  description='Replace substrings of the same chars with one char in the following string: ',
  sampling_fn=generate_deduplicate_data,
  fn=deduplicate,
  unary=True)

TASK_REGISTRY['length'] = Task(
  name='length',
  description='Return the length of the following string: ',
  sampling_fn=generate_random_str_given_range,
  fn=length,
  unary=True)

TASK_REGISTRY['longest_word'] = Task(
  name='longest_word',
  description='Return the longest word in the following string: ',
  sampling_fn=generate_longest_word_data,
  fn=longest_word,
  unary=True)

TASK_REGISTRY['duplicate'] = Task(
  name='duplicate',
  description='Duplicate each char in the following string: ',
  sampling_fn=generate_random_str_given_range,
  fn=duplicate,
  unary=True)



### Basic binary functions ###

def count(inputs):
  seq, query = inputs
  counter = collections.Counter(seq.split(' '))
  count = counter[query]
  return _str_to_data(str(count))

def delete(inputs):
  seq, query = inputs
  seq_str = _data_to_str(seq)
  query_str = _data_to_str(query)
  if query_str not in seq_str or query_str == '':
    return seq
  else:
    query_len = len(query_str)
    query_index = seq_str.index(query_str)
    seq_str = seq_str[:query_index] + seq_str[query_index + query_len:]
    return _str_to_data(seq_str)


def filt(inputs):
  seq, query = inputs
  seq_str = _data_to_str(seq)
  query_str = _data_to_str(query)
  if query_str not in seq_str or query_str == '':
    return seq
  else:
    result = delete(inputs)
    return filt((result, query))


def search(inputs):
  seq, query = inputs
  seq_str = _data_to_str(seq)
  query_str = _data_to_str(query)
  if query_str in seq_str:
    return _str_to_data('yes')
  else:
    return _str_to_data('no')


def get_index(inputs):
  seq, query = inputs
  seq_str = _data_to_str(seq)
  query_str = _data_to_str(query)
  if query_str in seq_str:
    query_index = seq_str.index(query_str)
    return _str_to_data(str(query_index))
  else:
    return _str_to_data('-1')


def sort(inputs):
  seq, query = inputs
  seq_list = seq.split(' ')
  query_list = query.split(' ')
  char_order_dic = {char: order for order, char in enumerate(query_list)}
  seq_list = sorted(seq_list, key=lambda char: char_order_dic[char])
  return ' '.join(seq_list)


def replace_many(inputs):
  seq, query = inputs
  seq_list = seq.split(' ')
  query_list = query.split(' ')

  replace_dic = {}
  for i in range(0, len(query_list), 2):
    replace_dic[query_list[i]] = query_list[i+1]
  result = []
  for char in seq_list:
    if char in replace_dic:
      result.append(replace_dic[char])
    else:
      result.append(char)
  return ' '.join(result)


def replace(inputs):
  return replace_many(inputs)


def union(inputs):
  seq1, seq2 = inputs
  seq1_set = set(seq1.split(' '))
  seq2_set = set(seq2.split(' '))
  result_set = seq1_set.union(seq2_set)
  return ' '.join(list(result_set))


def intersect(inputs):
  seq1, seq2 = inputs
  seq1_set = set(seq1.split(' '))
  seq2_set = set(seq2.split(' '))
  result_set = seq1_set.intersection(seq2_set)
  return ' '.join(list(result_set))


def set_1_minus_2(inputs):
  seq1, seq2 = inputs
  seq1_set = set(seq1.split(' '))
  seq2_set = set(seq2.split(' '))
  result_set = seq1_set - seq2_set
  return ' '.join(list(result_set))


def set_2_minus_1(inputs):
  seq1, seq2 = inputs
  seq1_set = set(seq1.split(' '))
  seq2_set = set(seq2.split(' '))
  result_set = seq2_set - seq1_set
  return ' '.join(list(result_set))


TASK_REGISTRY['count'] = Task(
  name='count',
  description='Count the char used in the following string: ',
  sampling_fn=generate_count_data,
  fn=count,
  unary=False)

TASK_REGISTRY['delete'] = Task(
  name='delete',
  description='Delete first appearance of a query string in the string: ',
  sampling_fn=generate_delete_data,
  fn=delete,
  unary=False)

TASK_REGISTRY['filter'] = Task(
  name='filter',
  description='Delete all appearances of a query string in the string: ',
  sampling_fn=generate_filter_data,
  fn=filt,
  unary=False)

TASK_REGISTRY['get_index'] = Task(
  name='get_index',
  description='Get index of first appearance of a query string in the string: ',
  sampling_fn=generate_get_index_data,
  fn=get_index,
  unary=False)

TASK_REGISTRY['search'] = Task(
  name='search',
  description='Return \'yes\' if query string is in string else \'no\': ',
  sampling_fn=generate_search_data,
  fn=search,
  unary=False)

TASK_REGISTRY['sort'] = Task(
  name='sort',
  description='Return seq sorted based on order of elements in query: ',
  sampling_fn=generate_sort_data,
  fn=sort,
  unary=False)

TASK_REGISTRY['replace'] = Task(
  name='replace',
  description='Return seq with one element value replaced according to query: ',
  sampling_fn=generate_replace_data,
  fn=replace,
  unary=False)

TASK_REGISTRY['replace_many'] = Task(
  name='replace_many',
  description='Return seq with many element values replaced according to query: ',
  sampling_fn=generate_replace_many_data,
  fn=replace_many,
  unary=False)

TASK_REGISTRY['union'] = Task(
  name='union',
  description='Return set(seq1) union set(seq2): ',
  sampling_fn=generate_two_seq_data,
  fn=union,
  unary=False)

TASK_REGISTRY['intersect'] = Task(
  name='intersect',
  description='Return set(seq1) intersect set(seq2): ',
  sampling_fn=generate_two_seq_data,
  fn=intersect,
  unary=False)

TASK_REGISTRY['set_1_minus_2'] = Task(
  name='set_1_minus_2',
  description='Return set(seq1) - set(seq2): ',
  sampling_fn=generate_two_seq_data,
  fn=set_1_minus_2,
  unary=False)

TASK_REGISTRY['set_2_minus_1'] = Task(
  name='set_2_minus_1',
  description='Return set(seq2) - set(seq1): ',
  sampling_fn=generate_two_seq_data,
  fn=set_2_minus_1,
  unary=False)


### LIME ### 

from lime_datagen.generate_data import gen_one_lime_data

TASK_REGISTRY['lime_induct'] = Task(
  name='lime_induct',
  description='LIME induction task',
  sampling_fn=gen_one_lime_data,
  fn=None,
  unary=None)


TASK_REGISTRY['lime_deduct'] = Task(
  name='lime_deduct',
  description='LIME deduction task',
  sampling_fn=gen_one_lime_data,
  fn=None,
  unary=None)

TASK_REGISTRY['lime_abduct'] = Task(
  name='lime_abduct',
  description='LIME abduction task',
  sampling_fn=gen_one_lime_data,
  fn=None,
  unary=None)


### Summarization tasks ###
SUMMARIZATION_TASKS = ['summ_CopyKeywordOneSentence',
                        'summ_CopyKeywordMultipleSentences',
                        'summ_CopyKeywordMultipleSentencesShuffled',
                        'summ_CopyKeywordMultipleSentencesSorted',
                        'summ_CopyQuoted',
                        'summ_CopyBulleted',
                        'summ_ClassifyKeyword',
                        'summ_ReplaceClassKeyword',
                        'summ_MajorityKeyword',
                        'summ_TopicSegregation',
                        'summ_ThresholdNumber',
                        'summ_LargestNumber',
                        'summ_ParaphraseWords',
                        'summ_JoinClauses',
                        'summ_BreakClauses',
                        'summ_TruncateSentence']
from summarization_tasks_datagen.ourtasks import gen_one_summ_data
for task in SUMMARIZATION_TASKS:
  TASK_REGISTRY[task] = Task(
    name=task,
    description=task,
    sampling_fn=gen_one_summ_data,
    fn=None,
    unary=None)


### MBPP tasks ###

MBPP_TASKS = ['find_rotations', 'remove_fl_occ', 'count_common', 
  'split_lowerstring', 'rearange_string', 'merge_dictionaries_three']


def gen_mbpp_data(task, length_range, char_set):
  if length_range[0] == 0:
    length_range = (1, length_range[1])
  if task.name in ['find_rotations', 'split_lowerstring', 'rearange_string']:
    s = gen_mbpp_arg_str(task, length_range, char_set)
    input_data = _str_to_data(s)
    output = task.fn(s)
    if task.name == 'split_lowerstring':
       output = str(output).replace(" ", '').replace('[', '').replace(']', '').replace('\'','')
    output_data = _str_to_data(str(output))
  elif task.name == 'merge_dictionaries_three':
    d1, d2, d3 = gen_mbpp_arg_3dic(length_range, char_set)
    input_str = (str(d1) + '<sep>' + str(d2) + '<sep>' + str(d3)).replace(' ', '')\
                                        .replace('{', '').replace('}', '').replace('\'','')
    while len(input_str) < length_range[0] or len(input_str) > length_range[1]:
      d1, d2, d3 = gen_mbpp_arg_3dic(length_range, char_set)
      input_str = (str(d1) + '<sep>' + str(d2) + '<sep>' + str(d3)).replace(' ', '')\
                                        .replace('{', '').replace('}', '').replace('\'','')
    input_data = ' <sep> '.join([_str_to_data(s) for s in input_str.split('<sep>')])
    output = task.fn(d1, d2, d3)
    output_data = _str_to_data(str(output).replace(' ', '').replace('{', '').replace('}', '').replace('\'',''))
  elif task.name == 'remove_fl_occ':
    s, c  = gen_mbpp_arg_str_arg_ch(length_range, char_set)
    input_data = _str_to_data(s) + ' <sep> ' + c
    output = task.fn(s, c)
    output_data = _str_to_data(output)
  elif task.name == 'count_common':
    words = gen_mbpp_arg_words(task, length_range, char_set)
    input_str = ''.join(words)
    while len(input_str) < length_range[0] or len(input_str) > length_range[1]:
      words = gen_mbpp_arg_words(length_range, char_set)
      input_str = ''.join(words)
    input_data = _str_to_data(input_str)
    output = str(task.fn(words)).replace(')', '').replace(')','').replace('(', '')\
                        .replace(" ", '').replace('[', '').replace(']', '').replace('\'','')
    output_data = _str_to_data(output)
  else:
    raise NotImplementedError
  return input_data, output_data


import string
def gen_mbpp_arg_str(task, length_range, char_set):
  if task.name == 'split_lowerstring':
    # check there is lowercase in charset
    lowercase_letters = set([x for x in string.ascii_lowercase])
    if len(char_set.union(lowercase_letters)) == 0:
      raise RuntimeError("must provide lowercase letters in char_set for split_lowerstring task")
  str_len = random.randint(*length_range)
  return ''.join(random.choices(list(char_set), k=str_len))


def gen_mbpp_arg_str_arg_ch(length_range, char_set):
  chars = list(char_set)
  assert length_range[1] >= 3
  length_range = (length_range[0], length_range[1] - 2)
  str_len = random.randint(*length_range)
  return ''.join(random.choices(chars, k=str_len)), random.choice(chars)


def gen_mbpp_arg_words(task, length_range, char_set):
  if task.name == 'count_common':
    max_non_comma_chars = length_range[1] - int(length_range[1] / 5)
    non_comma_chars = random.randint(1, max_non_comma_chars)

    num_unique_words = random.randint(1, max(int(non_comma_chars/3),1))
    num_max_total_chars_each_word = _random_partition_no_zeros(non_comma_chars, num_unique_words)
    num_occurences_each_word = [random.randint(1,max(int(num/5),1)) for num in num_max_total_chars_each_word]
    length_each_word = [int(x/y) for x,y in zip(num_max_total_chars_each_word, num_occurences_each_word)]
    words = []
    for length, num_occurences in zip(length_each_word, num_occurences_each_word):
      words += [''.join(random.choices(list(char_set), k=length))]*num_occurences
    random.shuffle(words)
    return words
  else:
    raise NotImplementedError()


def gen_mbpp_arg_3dic(length_range, char_set):
  non_punctuation_chars = random.randint(6, max(int(length_range[1] * 3 / 4),6))
  num_unique_words = random.randint(6, max(int(non_punctuation_chars/3),6))
  num_max_total_chars_each_word = _random_partition_no_zeros(non_punctuation_chars, num_unique_words)
  num_occurences_each_word = [random.randint(1,num) for num in num_max_total_chars_each_word]
  length_each_word = [int(x/y) for x,y in zip(num_max_total_chars_each_word, num_occurences_each_word)]
  words = []
  dics = [{}, {}, {}]
  for length, num_occurences in zip(length_each_word, num_occurences_each_word):
    words += [''.join(random.choices(list(char_set), k=length))]*num_occurences
  if len(words) % 2 == 1:
    words = words[:-1]
  keys = words[:int(len(words)/2)]
  values = words[int(len(words)/2):]
  random.shuffle(values)
  dics[0][keys[0]] = values[-1]
  dics[1][keys[1]] = values[-2]
  dics[2][keys[2]] = values[-3]
  for k, v in zip(keys, values):
    which_dic = random.randint(0,2)
    dics[which_dic][k] = v
  random.shuffle(dics)
  return dics[0], dics[1], dics[2]


def find_rotations(str): 
    tmp = str + str
    n = len(str) 
    for i in range(1,n + 1): 
        substring = tmp[i: i+n] 
        if (str == substring): 
            return i 
    return n 

  
def remove_fl_occ(s,ch): 
    for i in range(len(s)): 
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    for i in range(len(s) - 1,-1,-1):  
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    return s 


from collections import Counter
def count_common(words):
  word_counts = Counter(words)
  top_four = word_counts.most_common(4)
  return (top_four)


import collections as ct
def merge_dictionaries_three(dict1, dict2, dict3):
    merged_dict = dict(ct.ChainMap({},dict1,dict2,dict3))
    return merged_dict

import heapq
from collections import Counter
def rearange_string(S):
    ctr = Counter(S)
    heap = [(-value, key) for key, value in ctr.items()]
    heapq.heapify(heap)
    if (-heap[0][0]) * 2 > len(S) + 1: 
        return ""
    ans = []
    while len(heap) >= 2:
        nct1, char1 = heapq.heappop(heap)
        nct2, char2 = heapq.heappop(heap)
        ans.extend([char1, char2])
        if nct1 + 1: heapq.heappush(heap, (nct1 + 1, char1))
        if nct2 + 1: heapq.heappush(heap, (nct2 + 1, char2))
    return "".join(ans) + (heap[0][1] if heap else "")


import re
def split_lowerstring(text):
 return (re.findall('[a-z][^a-z]*', text))


for task in MBPP_TASKS:
  TASK_REGISTRY[task] = Task(
    name=task,
    description=task,
    sampling_fn=gen_mbpp_data,
    fn=eval(task),
    unary=None)
