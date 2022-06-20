import sys
import numpy as np


path = sys.argv[1]
n = sys.argv[2]
metrics_to_print = sys.argv[3].split(',')
def get_steps_and_metrics(path, eval_freq=0, is_offset=False, mod=-1):
    steps = []
    metrics = {}
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


s, m = get_steps_and_metrics(path)
n = [-int(x) for x in n.split(',')]
for metric in m.keys():
    if metric not in metrics_to_print:
        continue
    print("\n")
    print(f"=================={path}")
    print(metric)
    if n[0] == 0 and n[1] == 0:
        # for i in m[metric]:
        #     print(i)
        print(m[metric])
    else:
        print(f'printing mean ={metric}= index -{n[0]} to -{n[1]}')
        if n[1] == 0:
            print(m[metric])
            print(np.mean(m[metric][n[0]:]))
        else:
            print(m[metric])
            print(np.mean(m[metric][n[0]:n[1]]))

