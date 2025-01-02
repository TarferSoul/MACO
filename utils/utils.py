import math
import random

import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def compute_t(d, pi, phase, M, K, T, delta):
    # compute the number of rounds add / 10
    return 1 * d * pi * (2**(1 / 2 * phase)) * np.log(M * K * np.log(T) / delta)


def find_optimal_design_distr(arm_martix, max_iter_=10000, tol=1e-6):
    # find the optimal design distribution
    # arm_martix: a matrix of arms (each row is an arm)
    # tol: tolerance
    # return: optimal design distribution
    points = np.asmatrix(arm_martix)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    u = np.ones(N) / N
    return u 
    err = 1 + tol
    i = 0
    while err > tol:  # or not all(u > 0):
        X = Q * np.diag(u) * Q.T
        try:
            M = np.diag(Q.T * np.linalg.inv(X) * Q)
        except:
            print("Error: Singular Matrix, try to use pseudo inverse")
            M = np.diag(Q.T * np.linalg.pinv(X) * Q)
        jdx = np.argmax(M)
        step_size = 0.1 * (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u - u)
        # if i % 50 == 0:
        u = new_u
        i += 1
        if i > max_iter_:
            print("Maximum number of iterations reached")
            break
    return u
