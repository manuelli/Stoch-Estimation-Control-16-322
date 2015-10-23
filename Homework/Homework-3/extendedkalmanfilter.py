__author__ = 'manuelli'


#add support for dummy estimator

class Estimator:

    def __init__(self, A, B, W, R, dt, h, grad_h, useDummyEstimator=False):
        self.A = A
        self.x_dim = np.shape(A)[1]
        self.B = B
        self.W = W
        self.R = R
        self.h = h
        self.dt = dt
        self.Bdt = dt*B
        self.grad_h = grad_h

        self.useDummyEstimator = useDummyEstimator

    def initialize(self,x_0, Q_0):
        self.x_current = x_0
        self.Q_current = Q_0

    def update(self, u, y):

        x_process = np.dot(self.A, self.x_current) + np.dot(self.Bdt, u)
        Q_process = np.dot(self.A, self.Q_current, self.A.transpose()) + self.W

        # this just propagates the estimate through the process model.
        # doesn't use any measurements
        if self.useDummyEstimator:
            self.x_current = x_process
            self.Q_current = Q_process
            return (self.x_current, self.Q_current)

        h_current = self.h(self.x_current)
        H = self.grad_h(self.x_current) # gradient of measurement function at current estimate

        # measurement minus current mean
        y_bar = y - h_current

        # now we can compute the Kalman gain
        # K = Q_current * H^T [ H Q_current H^T + R]^{-1}
        Q_current_H_T = np.dot(self.Q_current, H.transpose())
        HQH_R_inv = np.linalg.inv(np.dot(np.dot(H, self.Q_current), H.transpose()) + self.R)
        K = np.dot(Q_current_H_T, HQH_R_inv)

        I = np.identity(self.x_dim)
        Q_new = np.dot(I - np.dot(K, H), self.Q_current)
        x_new = self.x_current + np.dot(K, y - h_current)

        # bookkeeping
        self.Q_current = Q_new
        self.x_current = x_new

        return (x_new, Q_new)

