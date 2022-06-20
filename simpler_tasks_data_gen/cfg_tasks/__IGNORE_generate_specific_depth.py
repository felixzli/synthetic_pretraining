# Natural Language Toolkit: Generating from a CFG
#
# Copyright (C) 2001-2022 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#     Peter Ljunglöf <peter.ljunglof@heatherleaf.se>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

import itertools
import sys

from nltk.grammar import Nonterminal


DEPTH_REACHED_STRING = 'D_R'

def generate_specific_depth(grammar, n, start=None, depth=None):
  """
  Generates n sentences from from grammar with specific depth 

  :param grammar: The Grammar used to generate sentences.
  :param n: The maximum number of sentences to return.

  :param start: The Nonterminal from which to start generate sentences.
  :param depth: The maximal depth of the generated tree.
  :return: An iterator of lists of terminal tokens.
  """
  if not start:
    start = grammar.start()
  if depth is None:
    depth = sys.maxsize
  iter = _generate_all(grammar, [start], depth)

  iter = itertools.islice(iter, None)
  
  specific_depth_sentences = []
  for candidate in iter:
    if DEPTH_REACHED_STRING in candidate:
      candidate = list(filter(lambda x: x != DEPTH_REACHED_STRING, candidate))
      specific_depth_sentences.append(candidate)
    if len(specific_depth_sentences) > n:
      break
  return specific_depth_sentences



def _generate_all(grammar, items, depth):
  if items:
    try:
      for frag1 in _generate_one_specific_depth(grammar, items[0], depth):
        for frag2 in _generate_all(grammar, items[1:], depth):
          yield frag1 + frag2
    except RecursionError as error:
      # Helpful error message while still showing the recursion stack.
      raise RuntimeError(
        "The grammar has rule(s) that yield infinite recursion!"
      ) from error
  else:
    yield []


def _generate_one_specific_depth(grammar, item, depth):
  if depth > 0:
    if isinstance(item, Nonterminal):
      for prod in grammar.productions(lhs=item):
        yield from _generate_all(grammar, prod.rhs(), depth - 1)
    else:
      if depth == 1:
        yield [item, DEPTH_REACHED_STRING]
      else:
        yield [item]
      # elif depth == 1:
      #   yield [item]
  # print('lol')


# def _generate_one(grammar, item, depth):
#   if depth > 0:
#     if isinstance(item, Nonterminal):
#       for prod in grammar.productions(lhs=item):
#         yield from _generate_all(grammar, prod.rhs(), depth - 1)
#     else:
#       yield [item]





demo_grammar = """
  S -> NP VP
  NP -> Det N
  PP -> P NP
  VP -> 'slept' | 'saw' NP | 'walked' PP
  Det -> 'the' | 'a'
  N -> 'man' | 'park' | 'dog'
  P -> 'in' | 'with'
"""


def demo(N=23):
  from nltk.grammar import CFG

  print("Generating the first %d sentences for demo grammar:" % (N,))
  print(demo_grammar)
  grammar = CFG.fromstring(demo_grammar)
  for n, sent in enumerate(generate(grammar, n=N), 1):
    print("%3d. %s" % (n, " ".join(sent)))
