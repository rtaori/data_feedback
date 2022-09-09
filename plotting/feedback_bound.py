import numpy as np


'''
Computes Theorem 1 upper bound
t: number of feedback rounds, t >= 0
n0: initial training set size
m: number of human-labeled samples per round
k: number of model-labeled samples per round
delta_n0: calibration error of the initial model
'''
def theorem1_bound(t, n0, m, k, delta_n0):
    n = lambda s: n0 + s * (m + k)
    val = 0
    for i in range(1, t+1):
        val += k / n(i) * np.prod([(n(j) - m) / n(j) for j in range(i+1, t+1)])
    val = (1 + val) * delta_n0
    return val

'''
Computes Theorem 1 upper bound for many rounds
T: total number of feedback rounds to provide bound for, T >= 0
n0: initial training set size
m: number of human-labeled samples per round
k: number of model-labeled samples per round
delta_n0: calibration error of the initial model
returns: theorem 1 upper bound for 0 <= t <= T
'''
def theorem1_continual_bound(T, n0, m, k, delta_n0):
    return np.array([theorem1_bound(t, n0, m, k, delta_n0) for t in range(T+1)])
