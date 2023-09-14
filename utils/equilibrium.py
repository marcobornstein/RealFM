import numpy as np
import scipy


def accuracy(x, a_opt, k):
    return np.maximum(0, a_opt - ((np.sqrt(2*k*(2 + np.log(x/k))) + 4) / np.sqrt(x)))


def accuracy_utility(acc, a, b):
    return a/np.power((1-acc), b)


def utility(num_data, cost, k, b, a_opt=0.95, a=1):
    acc = accuracy(num_data, a_opt, k)
    u = accuracy_utility(acc, a, b) - cost*num_data - a
    return u


def inverse_utility(num_data, cost, k, b, a_opt=0.95, a=1):
    return -utility(num_data, cost, k,  b, a_opt=a_opt, a=a)


def old_utility(num_data, cost, k, a_opt=0.95):
    return accuracy(num_data, a_opt, k) - cost*num_data


def inverse_old_utility(num_data, cost, k, a_opt=0.95):
    return -old_utility(num_data, cost, k, a_opt=a_opt)


def optimal_data_local(cost, b=2, k=1, linear=False):
    x_init = np.array(1e4)
    if not linear:
        sol = scipy.optimize.fmin(inverse_utility, x_init, args=(cost, k, b))[0]
        util = utility(sol, cost, k, b)
    else:
        sol = scipy.optimize.fmin(inverse_old_utility, x_init, args=(cost, k))[0]
        util = old_utility(sol, cost, k)

    if util <= 0:
        num_data = 0
    else:
        num_data = int(sol)

    return num_data
