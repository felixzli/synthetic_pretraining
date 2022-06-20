import os.path as osp


def get_filename_no_ext(path):
  return osp.basename(osp.splitext(path)[0])


def get_path_no_ext(path):
  return osp.splitext(path)[0]


def get_path_with_new_ext(path, ext):
  assert '.' in ext
  path_no_ext = get_path_no_ext(path)
  path = path_no_ext + ext
  return path


def make_path_with_prefix_added_to_basename(path, prefix):
  bn = osp.basename(path)
  new_bn = prefix + bn
  return osp.join(osp.dirname(path), new_bn)


import glob
def count_lines(path):
  pathss = glob.glob(path, recursive=True)
  print(f"========counting lines in {pathss}")
  count = 0
  for path in pathss:
    print(path)
    with open(path, mode='r') as f:
      for _ in f:
        count += 1

  print(f"========counting lines in {pathss} ==== {count}")
  return count


if __name__ == '__main__':
  path = 'heraldedbell/basictheseus.py'
  print(get_filename_no_ext(path))
  print(get_path_no_ext(path))
  print(get_path_with_new_ext(path, '.txt'))