import numpy as np
import scipy


def accuracy(x, a_opt, k):
    return np.maximum(0, a_opt - ((np.sqrt(2*k*(2 + np.log(x/k))) + 4) / np.sqrt(x)))


def accuracy_utility(acc, a, b):
    return a/np.power((1-acc), b) - a


def accuracy_utility_dx(acc, a, b, c):
    return (c*a*b)/np.power((1-acc), b+1)


def accuracy_utility_ddx(acc, a, b, c):
    return (c*a*b*(b+1))/np.power((1-acc), b+2)


def utility(num_data, cost, k, b, c, a_opt=0.95, a=1):
    acc = accuracy(num_data, a_opt, k)
    u = c*accuracy_utility(acc, a, b) - cost*num_data
    return u


def inverse_utility(num_data, cost, k, b, c, a_opt=0.95, a=1):
    return -utility(num_data, cost, k,  b, c, a_opt=a_opt, a=a)


def old_utility(num_data, cost, k, a_opt=0.95):
    return accuracy(num_data, a_opt, k) - cost*num_data


def inverse_old_utility(num_data, cost, k, a_opt=0.95):
    return -old_utility(num_data, cost, k, a_opt=a_opt)


def accuracy_shaping(b, b_local, a_bar, mc, c, linear=False, eps=1e-9):
    if linear:
        gamma = (mc+eps)*(b-b_local)
    else:
        phi_dx = accuracy_utility_dx(a_bar, 1, 2, c)
        phi_ddx = accuracy_utility_ddx(a_bar, 1, 2, c)
        gamma = (-phi_dx + np.sqrt(phi_dx**2 + (2*phi_ddx)*(mc+eps)*(b-b_local)))/phi_ddx
    return gamma


def accuracy_shaping_max(b, b_local, a_bar, a_fed, mc, c, linear=False):
    acc_diff = a_fed - (a_bar + accuracy_shaping(b, b_local, a_bar, mc, c, linear))
    return acc_diff


def optimal_data_local(cost, b=2, k=1, linear=False, c=1, a_opt=0.95):
    x_init = np.array(1e4)
    if not linear:
        sol = scipy.optimize.fmin(inverse_utility, x_init, args=(cost, k, b, c, a_opt))[0]
        util = utility(sol, cost, k, b, c, a_opt=a_opt)
    else:
        sol = scipy.optimize.fmin(inverse_old_utility, x_init, args=(cost, k, a_opt))[0]
        util = old_utility(sol, cost, k, a_opt=a_opt)

    if util <= 0:
        num_data = 0
    else:
        num_data = int(sol)

    return num_data


def optimal_data_fed(a_local, a_fed, b_local, mc, c=1, linear=False):
    sol = scipy.optimize.root(accuracy_shaping_max, np.array(b_local), args=(b_local, a_local, a_fed, mc, c, linear))
    return int(sol.x)
