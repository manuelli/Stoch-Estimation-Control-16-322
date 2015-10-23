__author__ = 'manuelli'
import numpy as np

class Controller:

    # x_d, u_d are functions which define the desired trajectory and control inputs.
    # K is the controller gain
    def __init__(self, x_des, u_des, K):
        self.x_des = x_des
        self.u_des = u_des
        self.K = K

    def computeControlInput(self, x, t):
        return self.u_des(t) - np.dot(self.K, x - self.x_des(t))
