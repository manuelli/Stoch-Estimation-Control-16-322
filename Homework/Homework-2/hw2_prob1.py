__author__ = 'manuelli'
import numpy as np

z_data = np.array([30.1, 45.0, 74.6])
z_data = np.deg2rad(z_data)
l_data = np.array([0,500,1000])
r_data = np.array([0.01,0.01,0.04])
deg2rad = 2*np.pi/360.0
r_data = deg2rad**2*r_data
R = np.diag(r_data)
R_inv = np.linalg.inv(R)

def h(x,l):
    return np.arctan(x[1]/(x[0] - l))

def grad_h(x,l):
    dh = 1.0/(1.0 + (x[1]/(x[0] - l))**2)

    dx_0 = -x[1]/(x[0] - l)**2
    dx_1 = 1.0/(x[0] - l)

    grad = np.array([dh*dx_0, dh*dx_1])
    return grad

def h_vec(x):
    hVec = np.zeros(np.size(z_data))

    for i in xrange(0, np.size(hVec)):
        hVec[i] = h(x,l_data[i])

    return hVec

def grad_h_matrix(x):
    N = np.size(z_data)
    gradH = np.zeros((N,2))

    for i in xrange(0,N):
        gradH[i,:] = grad_h(x,l_data[i])

    return gradH


def WSSE(x):
    diff = z_data - h_vec(x)
    diff = diff[:,None] # makes it into an N x 1 vector

    wsse = np.dot(np.dot(diff.transpose(),R_inv),diff)

    # should return a scalar
    return wsse


def nllsUpdate(x_current):
    z_tilde = z_data - h_vec(x_current)
    H = grad_h_matrix(x_current)


    HRH = np.dot(np.dot(H.transpose(), R_inv), H)
    HTR_inv = np.dot(H.transpose(),R_inv)
    dx = np.dot(np.dot(np.linalg.inv(HRH), HTR_inv), z_tilde)
    x_new = x_current + dx
    return x_new

def nlls(x_initial, tol=1e-4, verbose=False, maxNumIterations=1000):
    x_current = x_initial
    wsse_old = WSSE(x_initial)
    eps = 1
    numIterations = 0

    while (eps > tol):
        x_current = nllsUpdate(x_current)
        wsse_current = WSSE(x_current)
        eps = np.abs(wsse_current - wsse_old)
        wsse_old = wsse_current
        numIterations += 1

        if (numIterations > maxNumIterations):
            print "reached max num iterations"
            break

    H = grad_h_matrix(x_current)
    Sigma = np.linalg.inv(np.dot(np.dot(H.transpose(),R_inv),H))

    if verbose:
        print "  "
        print "---NLLS Summary---"
        print "x = " + str(x_current)
        print "objective value = " + str(wsse_old)
        print "num iterations = " + str(numIterations)
        print "uncertainty matrix = "
        print Sigma

    return x_current

if __name__=="__main__":
    x_initial = np.array([1200,1200])
    nlls(x_initial, verbose=True)






