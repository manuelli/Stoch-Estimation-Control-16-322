__author__ = 'manuelli'
import numpy as np
import matplotlib.pyploy as plt
form scipy.stats import norm


def simulate(B, N=100, Sigma=np.eye(2),var=1):
    y1 = np.zeros(N)
    y2 = np.zeros(N)