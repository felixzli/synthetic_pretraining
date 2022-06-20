import pathlib


def parent_dir(path):
  path = pathlib.Path(path) 
  parent_dir = str(path.parent)
  if 'gs:/' in parent_dir and 'gs://' not in parent_dir:
    return parent_dir.replace('gs:/', 'gs://')
  else:
    return parent_dir