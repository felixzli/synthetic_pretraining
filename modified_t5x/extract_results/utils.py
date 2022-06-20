def get_steps_and_metrics(path, eval_freq=0, # eval_freq arg only required if is_offset is True
                      is_offset=False, mod=-1):
  if is_offset:
    assert eval_freq != 0 
  steps = []
  metrics = {}
  if not is_offset:
    offset = 0
  with open(path, mode='r') as f:
    for i, l in enumerate(f):
      l = eval(l)
      s = l['step']
      metric_names = list(set(l.keys()) - {'step'})
      if i == 0:
        for mn in metric_names:
          metrics[mn] = []
      if mod == -1 or s % mod == 0:
        if i == 0:
          offset = (s - eval_freq)
        if is_offset:
          steps.append(s - offset)
        else:
          steps.append(s)
        for mn in metric_names:
          metrics[mn].append(l[mn])
  return steps, metrics

