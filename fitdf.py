from __future__ import division
import pickle
import numpy as np
from scipy import optimize

np.random.seed(102)

T = 1e-3
tau_s = 1e-4
fs = 1e6
dt = 1./fs
N = int(round(T*fs))

t = np.arange(N)*dt


