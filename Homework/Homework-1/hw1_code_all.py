__author__ = 'manuelli'
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

#####################
# Problem 4 code

def problem3():
    S = np.array([[3,1],[1,2]])
    eig = np.linalg.eig(S)
    print "eigenvalues and eigenvectors are "
    print eig
    lam1 = eig[0][0]
    lam2 = eig[0][1]
    v1 = eig[1][:,0]
    v2 = eig[1][:,1]

    def plotEllipse(c):
        b = lambda x: np.sqrt(lam2*(c**2-x**2/lam1))
        a_grid = np.linspace(-c*np.sqrt(lam1),c*np.sqrt(lam1),100)

        b_vec = np.vectorize(b)
        b_pos = b_vec(a_grid)
        b_neg = -1.0*b_pos
        z_pos = np.outer(a_grid,v1) + np.outer(b_pos,v2)
        z_neg = np.outer(a_grid,v1) + np.outer(b_neg,v2)
        plt.plot(z_pos[:,0],z_pos[:,1], color='b')
        plt.plot(z_neg[:,0],z_neg[:,1], color='b')


    c_vals = [0.25, 1.0, 1.5]
    for c in c_vals:
        plotEllipse(c)
        prob = 1-np.exp(-c**2/2)
        print "the probability of being in the ellipse of size c = " + str(c) + " is"
        print prob

    plt.show()

def problem4():
    M = 200000
    N = 6
    x_matrix = 2*np.random.sample((M,N)) - 1
    # print x_matrix

    z_vals = np.sum(x_matrix, axis=1)
    # print z_vals

    K = 10.0
    grid = np.linspace(-N,N,K*N)
    count = np.zeros(np.size(grid))
    for idx in xrange(0,np.size(grid) - 1):
        lb = grid[idx]
        ub = grid[idx + 1]
        count[idx] = np.size(np.where((z_vals > lb) & (z_vals < ub)))/(1.0*M)*K/2.0

    var = N/3.0
    scale = np.sqrt(var)
    fine_grid = np.linspace(-N,N,100*N)
    pdf_points = norm.pdf(fine_grid,scale=scale)
    plt.plot(grid,count)
    # print np.shape(pdf_points)
    plt.plot(fine_grid,pdf_points,color='r')
    plt.show()


def problem5():

    # this is the pdf of Y
    def pdfY(x):
        # if (x<=0):
        #     return 0
        return np.divide(norm.pdf(np.sqrt(x)), np.sqrt(x))

    x = np.linspace(-3,3,200)
    x_pos = np.linspace(0.01,3,100)
    plt.plot(x, norm.pdf(x))
    plt.plot(x_pos, pdfY(x_pos))
    plt.ylim((0,2.0))
    plt.show()

# note: OLS doesn't work in this problem because we have non-exogeneous regressors.
# the resulting estimator will be biased and not consistent.
def problem7():
    B = np.array([[1,0],[0,0]])
    N = 1000
    Sigma_x = np.eye(2)
    mu_x = np.zeros(2)
    sigma_squared = 0.1
    X = np.random.multivariate_normal(mu_x,Sigma_x, size=N)
    eta_1 = np.random.multivariate_normal(mu_x,sigma_squared*np.eye(2), size=N)
    eta_2 = np.random.multivariate_normal(mu_x,sigma_squared*np.eye(2), size=N)
    y_1 = X + eta_1
    y_2 = np.dot(X,np.transpose(B)) + eta_2


    # plot the data
    plt.scatter(y_1[:,0],y_1[:,1],color='b', marker='o',facecolors='none')
    plt.scatter(y_2[:,0],y_2[:,1],color='r', marker='o',facecolors='none')

    # need to construct the H matrix
    H = np.zeros((2*N,4))

    y_1_stack = np.reshape(y_1,(2*N,))
    y_2_stack = np.reshape(y_2,(2*N,))


    H = np.zeros((2*N,4))
    for i in xrange(0,N):
        idx = 2*i;
        H[idx,0:2] = y_1[i,:]
        H[idx+1,2:4] = y_1[i,:]


    # print np.shape(H)
    A = np.linalg.inv(np.dot(H.transpose(), H))
    b = np.dot(np.dot(A,H.transpose()),y_2_stack)
    B = np.reshape(b,(2,2))
    print "the OLS estimate of B is, note that this estimator is biased due to lack of exogeneity "
    print B
    plt.show()

if __name__=="__main__":
    # call the function for the appropriate problem, e.g. 7 in this case
    problem7()

