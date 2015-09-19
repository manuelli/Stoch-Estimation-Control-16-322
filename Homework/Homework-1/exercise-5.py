__author__ = 'manuelli'

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

def pdfY(x):
    # if (x<=0):
    #     return 0
    return 1/2.0*np.divide(norm.pdf(x), 2*np.sqrt(x))

fig, ax = plt.subplots(1,1)
x = np.linspace(norm.ppf(0.01),norm.ppf(0.99),100)
x_pos = np.linspace(0.01,norm.ppf(0.99),100)
ax.plot(x, norm.pdf(x))
ax.plot(x_pos, pdfY(x_pos))
plt.show()