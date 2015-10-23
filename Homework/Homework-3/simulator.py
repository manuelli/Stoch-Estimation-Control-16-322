__author__ = 'manuelli'
import numpy as np
import matplotlib.pyplot as plt
import sys


class Simulator:

    # x_{k+1} = Ax_k + (B*dt) u_k + w
    # y_k = h(x_k) + v
    # h is the measurement function
    # grad_h is the gradient of the measurement function
    def __init__(self, dt, A, B, h, grad_h, W, R):
        self.A = A
        self.B = B
        self.h = h
        self.grad_h = grad_h
        self.dt = dt
        self.Bdt = dt*B
        self.estimator = None
        self.controller = None

        # this is the size of x
        self.x_dim = np.shape(A)[1]
        self.u_dim = np.shape(B)[1]

    def setEstimator(self, estimator):
        self.estimator = estimator

    def setController(self, controller):
        self.controller = controller

    def sample_w(self):
        return np.random.multivariate_normal(np.zeros(2), self.W)

    def sample_v(self):
        return np.random.multivariate_normal(np.zeros(2), self.R)

    def stepProcessModel(self, x, u, addNoise=True):
        x_new = np.dot(self.A,x) + np.dot(self.Bdt,u)

        if addNoise:
            x_new = x_new + self.sample_w()

        return x_new

    def generateMeasurement(self, x):
        return self.h(x) + self.sample_v()


    def simulate(self, x_0, Q_0, T):
        # initialize state estimator

        if (self.estimator is None):
            print "Missing Estimator: need an estimator to simulate"

        if (self.controller is None):
            print "Missing Controller: need a controller to simulate"

        t_grid = np.arange(0,T,dt)
        N = np.size(t_grid)

        # initialize state estimator
        self.estimator.initialize(x_0, Q_0)
        x_true = np.zeros((N,self.x_dim))
        x_est = np.zeros((N,self.x_dim))
        u_controlled = np.zeros((N,self.u_dim))
        Q_est = np.zeros((N,self.x_dim, self.x_dim))

        x_true[0] = x_0
        x_est[0] = x_0
        Q_est[0,:,:] = Q_0

        for idx in xrange(0,N-1):
            # t[idx] is current, t[idx+1] is next

            t = t_grid[idx]
            x_current_true = x_true[idx,:]
            x_current_est = x_est[idx, :]
            u = self.controller.computeControlInput(x_current_est,t)

            x_new_true = self.stepProcessModel(x_current_true, u, addNoise=False)
            x_true[idx+1, :] = x_new_true
            y = self.generateMeasurement(x_new_true)

            (x_new_est, Q_new) = self.estimator.update(u,y)
            x_est[idx+1,:] = x_new_est
            Q_est[idx+1,:,:] = Q_new


        # store the simulation results so we can plot them later if we want
        self.simResults = dict()
        self.simResults['x_true'] = x_true
        self.simResults['x_est'] = x_est
        self.simResults['t_grid'] = t_grid
        self.simResults['u_controlled'] = u_controlled




def x_desired(t, T):
    X_corners = np.array([[0,0],[5,0],[5,5],[0,5],[0,0]])
    if (t < 0) or (t > T):
        raise ValueError('t out of bounds')

    for i in xrange(0,4):
        if t < 1.0*i*T/4.0:
            idx = i

    s = (t-idx*T/4.0)/(T/4.0)
    x = (1-s)*X_corners[idx,:] + s*X_corners[idx+1,:]

    return x

def u_desired(t, T):

    U_vals= 5/(T/4.0)*np.array([[1,0],[0,1],[-1,0],[0,-1]])
    if (t < 0) or (t > T):
        raise ValueError('t out of bounds')

    for i in xrange(0,4):
        if t < 1.0*i*T/4.0:
            idx = i

    u = U_vals[idx,:]
    return u




