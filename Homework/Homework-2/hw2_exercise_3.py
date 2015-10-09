__author__ = 'manuelli'

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

f = -0.5
q_c = 1.0
T = 5.0

def propagateDistribution(dt = 0.05, mu_0 = 1, Q_0 = 2.0, showPlot=False):
    # first we need to compute the discrete time approximation to the continuous system
    S = np.array([[-f,q_c],[0,f]])
    C = scipy.linalg.expm2(dt*S)
    A_d = C[1,1]
    W_d = C[0,1]*C[1,1]

    print "A_d is " + str(A_d)
    print "W_d is " + str(W_d)

    t_vals = np.linspace(0,T,T/(dt*1.0))
    N = np.size(t_vals)
    mu = np.zeros(N)
    Q = np.zeros(N)


    mu[0] = mu_0
    Q[0] = Q_0

    for i in xrange(1,N):
        (mu[i],Q[i]) = propagateDistributionStep(A_d, W_d, mu[i-1], Q[i-1])

    if showPlot:
        plt.plot(t_vals, mu)
        plt.plot(t_vals, Q)
        # plt.show()

    return (t_vals, mu, Q)


def propagateDistributionStep(A_d, W_d, mu, Q):
    mu_next = A_d*mu
    Q_next = A_d*Q*A_d + W_d
    return (mu_next, Q_next)



if __name__ == "__main__":
    # this is the plot for Q_0 = 2
    propagateDistribution(Q_0 = 2.0, showPlot=True)
    propagateDistribution(Q_0 = 0, showPlot=True)
    plt.show()

