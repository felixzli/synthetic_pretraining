import os
import sys
import json
path = sys.argv[1]
n = sys.argv[2]

def print_inferences(path, n):
  save_file = open(f'extract_results/tmp/first_{n}_inferences.txt', mode='w')
  with open(path, mode='r') as f:
      for i, l in enumerate(f):
        if i == n:
          break
        l = eval(l)
        print(l['prediction'])
        print('\"' + l['prediction'] + '\"', file=save_file)

os.mkdir('extract_results/tmp/')
print_inferences(path, n)