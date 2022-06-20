import os


def wc_files(file_objs):
  for f in file_objs:
    os.system(f'wc -l {f.name} ')


def head_files(file_objs, n=2):
  for f in file_objs:
    print('======')
    print(f.name)
    print('---')
    os.system(f'head -n {n} {f.name}')
