import numpy as np
import scipy
import matplotlib.pyplot as plt


def accuracy(x, a_opt, k):
    return np.maximum(0, a_opt - ((np.sqrt(2*k*(2 + np.log(x/k))) + 4) / np.sqrt(x)))


def accuracy_utility(acc, a, b):
    return a/np.power((1-acc), b)


def utility(num_data, cost, k, a_opt=0.95, a=1, b=2):
    acc = accuracy(num_data, a_opt, k)
    u = accuracy_utility(acc, a, b) - cost*num_data - a
    return u


def inverse_utility(num_data, cost, k, a_opt=0.95, a=1, b=2):
    return -utility(num_data, cost, k, a_opt=a_opt, a=a, b=b)


def old_utility(num_data, cost, k, a_opt=0.95):
    return accuracy(num_data, a_opt, k) - cost*num_data


def inverse_old_utility(num_data, cost, k, a_opt=0.95):
    return -old_utility(num_data, cost, k, a_opt=a_opt)


def optimal_data_plot(x_init, k_vals, costs, savefig=True):
    maxs = np.empty_like(costs)
    maxs_old = np.empty_like(costs)
    for k in k_vals:
        for i, cost in enumerate(costs):

            max = scipy.optimize.fmin(inverse_utility, x_init, args=(cost, k))
            max_old = scipy.optimize.fmin(inverse_old_utility, x_init, args=(cost, k))

            val = utility(max, cost, k)
            if val > 0:
                maxs[i] = max
            else:
                maxs[i] = np.nan

            val_old = old_utility(max_old, cost, k)
            if val_old > 0:
                maxs_old[i] = max_old
            else:
                maxs_old[i] = np.nan

        label = 'k = ' + str(k) + ' (Power)'
        label_old = 'k = ' + str(k) + ' (Linear)'
        plt.plot(costs, maxs, label=label)
        plt.plot(costs, maxs_old, label=label_old)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Marginal Cost', fontsize=18, weight='bold')
    plt.ylabel('Data Contribution ($m_i$)', fontsize=18, weight='bold')
    plt.grid()

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [0, 2, 4, 1, 3, 5]

    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    if savefig:
        plt.savefig('marginal-cost-comparison.pdf')
    else:
        plt.show()


def utility_plots(cost, k=1, savefig=True):
    plt.figure()
    x = np.logspace(0, 7, 5001)
    u = utility(x, cost, k)
    u_old = old_utility(x, cost, k)
    plt.plot(x, u, label='Power Function $\phi_i$')
    plt.plot(x, u_old, label='Linear $\phi_i$')
    plt.yscale('symlog')
    plt.xscale('log')
    plt.ylabel('Utility', fontsize=18, weight='bold')
    # plt.xlabel('Data Contribution')
    plt.xlabel('Data Contribution ($m_i$)', fontsize=18, weight='bold')
    plt.legend(loc='lower left', fontsize=18)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    if savefig:
        title = 'utility-comparison-' + str(cost) + '.pdf'
        plt.savefig(title)
    else:
        plt.show()


if __name__ == '__main__':

    costs = np.logspace(-5, -1, 5001)
    k_vals = [1, 10, 100]
    x_init = np.array(1e4)

    optimal_data_plot(x_init, k_vals, costs, savefig=True)
    # utility_plots(0.0001, savefig=True)
