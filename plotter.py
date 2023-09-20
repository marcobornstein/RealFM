import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.equilibrium import accuracy_utility, accuracy
from textwrap import wrap


def unpack_data(directory_path, epochs, num_workers, datatype='fed-epoch-acc-top1.log'):
    directory = os.path.join(directory_path)
    if not os.path.isdir(directory):
        raise Exception(f"custom no directory {directory}")
    data = np.zeros((epochs, num_workers))
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(datatype):
                j = int(file.split('-')[0][1:])
                with open(directory_path + '/' + file, 'r') as f:
                    i = 0
                    for line in f:
                        itms = line.strip().split('\n')[0]
                        data[i, j] = float(itms)
                        i += 1
    return data


def bootstrapping(data, num_per_group, num_of_group):
    new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
    return new_data


def generate_confidence_interval(ys, number_per_g=30, number_of_g=1000, low_percentile=1, high_percentile=99):
    means = []
    mins =[]
    maxs = []
    for i, y in enumerate(ys.T):
        y = bootstrapping(y, number_per_g, number_of_g)
        means.append(np.mean(y))
        mins.append(np.percentile(y, low_percentile))
        maxs.append(np.percentile(y, high_percentile))
    return np.array(means), np.array(mins), np.array(maxs)


def plot_ci(x, y, num_runs, num_dots, mylegend, ls='-', lw=3, transparency=0.2):
    assert(x.ndim == 1)
    assert(x.size == num_dots)
    assert(y.ndim == 2)
    assert(y.shape == (num_runs, num_dots))
    y_mean, y_min, y_max = generate_confidence_interval(y)
    plt.plot(x, y_mean, 'o-', label=mylegend, linestyle=ls, linewidth=lw)  # , label=r'$\alpha$={}'.format(alpha))
    plt.fill_between(x, y_min, y_max, alpha=transparency)
    return


if __name__ == '__main__':
    colors = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']
    nw = 8
    epochs = 100
    fed_accs = []
    local_accs = []
    iters = np.arange(1, epochs + 1)
    each_dev_local_a = []

    # strictly accuracy plots
    '''
    for trial in range(1, 4):
        file = 'output/Cifar10/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        fed_test_acc = unpack_data(file, epochs, nw)
        local_test_acc = unpack_data(file, epochs, nw, datatype='local-epoch-acc-top1.log')
        fed_accs.append(fed_test_acc[:, 0])
        local_accs.append(np.mean(local_test_acc, axis=1))
        each_dev_local_a.append(local_test_acc[-1, :])

    # compute error bars over all three runs
    fed_accs = np.stack(fed_accs, axis=0)
    local_accs = np.stack(local_accs, axis=0)

    y_mean, y_min, y_max = generate_confidence_interval(fed_accs)
    y_mean_local, y_min_local, y_max_local = generate_confidence_interval(local_accs)

    plt.figure()

    # plot federated results
    plt.plot(iters, y_mean, label='Federated Training', color='b')
    plt.fill_between(iters, y_min, y_max, alpha=0.2, color='b')

    # plot average of local results
    plt.plot(iters, y_mean_local, label='Average Local Training', color='r')
    plt.fill_between(iters, y_min_local, y_max_local, alpha=0.2, color='r')

    title = 'Federated Training vs. Average Local Training for CIFAR-10'
    plt.title(title)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    plt.xlim([1, epochs])
    plt.xscale("log")
    # plt.ylim([0.225, 0.48])
    plt.grid(which="both", alpha=0.25)
    # plt.show()
    savefilename = str(nw) + 'device' + '.png'
    plt.savefig(savefilename, dpi=200)
    '''

    # =======================================================
    # plot data contribution plot and bar chart

    # fed_a = y_mean[-1] * np.ones(nw)
    nw = 8
    a_opt = 0.9
    k = 18
    local_b = np.ones((15, nw)) * np.nan
    fed_b = np.ones((15, nw)) * np.nan
    payoff = np.ones((15, nw)) * np.nan
    mc = np.ones((15, nw)) * np.nan
    x = ['RealFM (Non-Uniform Payoff and Cost)', 'RealFM (Non-Uniform Cost)', 'RealFM (Uniform)',
         'Linear RealFM (Non-Uniform Cost)', 'Linear RealFM (Uniform)', 'Local Training']

    for trial in range(1, 4):

        file = 'output/Cifar10/realfm-nonuniformPC-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial - 1, :] = optimal_data[0, :]
        payoff[trial - 1, :] = optimal_data[1, :]
        local_b[trial - 1, :] = optimal_data[2, :]
        fed_b[trial - 1, :] = optimal_data[3, :]

        file = 'output/Cifar10/realfm-nonuniformC-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 2, :] = optimal_data[0, :]
        payoff[trial + 2, :] = optimal_data[1, :]
        local_b[trial + 2, :] = optimal_data[2, :]
        fed_b[trial + 2, :] = optimal_data[3, :]

        file = 'output/Cifar10/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 5, :] = optimal_data[0, :]
        payoff[trial + 5, :] = optimal_data[1, :]
        local_b[trial + 5, :] = optimal_data[2, :]
        fed_b[trial + 5, :] = optimal_data[3, :]

        file = 'output/Cifar10/realfm-linear-nonuniformC-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 8, :] = optimal_data[0, :]
        payoff[trial + 8, :] = optimal_data[1, :]
        local_b[trial + 8, :] = optimal_data[2, :]
        fed_b[trial + 8, :] = optimal_data[3, :]

        file = 'output/Cifar10/realfm-linear-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 11, :] = optimal_data[0, :]
        payoff[trial + 11, :] = optimal_data[1, :]
        local_b[trial + 11, :] = optimal_data[2, :]
        fed_b[trial + 11, :] = optimal_data[3, :]

    # local_a = np.mean(np.stack(each_dev_local_a, axis=0), axis=0)
    local_b_mean = np.nanmean(local_b[:3], axis=0)
    fed_b_mean = np.nanmean(fed_b[:3], axis=0)

    # bar plot
    '''
    added_b = fed_b_mean - local_b_mean
    devices = np.arange(1, nw+1)
    plt.figure()
    plt.bar(devices, local_b_mean, label='Local Optimal Contributions $(b_i^o)$')
    bar = plt.bar(devices, added_b, bottom=local_b_mean, label='Incentivized Contributions $(b_i^* - b_i^o)$')
    plt.xlabel('Devices', fontsize=13)
    plt.ylabel('Update Contribution ($b_i$)', fontsize=13)
    plt.xticks(devices)
    plt.legend(loc='lower left', fontsize=10)
    # plt.ylim([0, 5500])

    for i, rect in enumerate(bar):
        val = 100*(added_b[i] / local_b_mean[i])
        height = rect.get_height() + local_b_mean[i]
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'+{val:.0f}%', ha='center', va='bottom', fontsize=6)

    savefilename2 = str(nw) + 'device-bar-mean-nonuniformPC' + '.png'
    plt.savefig(savefilename2, dpi=200)
    # plt.show()
    '''

    # bar chart idea: plot for each test type (including local baseline)
    # SERVER BAR CHART
    # both 1) average gradient contributions across devices and 2) server utility

    '''
    x = ['Non-Uniform Payoff & Cost', 'Non-Uniform Cost', 'Uniform',
         'Non-Uniform Cost', 'Uniform', 'Local Training']

    x = ['\n'.join(wrap(l, 15)) for l in x]

    # local baseline (expected)
    if nw == 16:
        expected_local_b = 3000
    elif nw == 8:
        expected_local_b = 5500

    num_baselines = len(x)
    trials = 3
    avg_data_fed = np.empty(num_baselines)
    avg_utility_fed = np.empty(num_baselines)
    avg_data_fed[-1] = expected_local_b
    a_local = accuracy(expected_local_b, a_opt, k)
    avg_utility_fed[-1] = accuracy_utility(a_local, 1, 2)

    ind1 = np.arange(3)
    ind2 = np.arange(3, 5)
    ind3 = 5

    width = 0.3
    for i in range(num_baselines-1):
        all_data_avg = np.average(np.sum(fed_b[trials*i:(trials*(i+1)), :], axis=1))
        total_fed = np.average(fed_b[trials*i:(trials*(i+1)), :], axis=1)
        avg_data_fed[i] = np.average(total_fed)

        # mechanism baseline
        acc_fed = accuracy(all_data_avg, a_opt, k)
        avg_utility_fed[i] = accuracy_utility(acc_fed, 1, 2)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # create axis labels
    ax1.set_xlabel('Training Methods', fontsize=13)
    ax1.set_ylabel('Server Utility', color='tab:red', fontsize=13)
    ax1.bar(ind1, avg_utility_fed[:3], width, color='tab:red')
    ax1.bar(ind2, avg_utility_fed[3:5], width, color='tab:red', hatch='//')
    ax1.bar(ind3, avg_utility_fed[-1], width, color='tab:red', hatch='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Adding Twin Axes
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Gradient Contributions Across Devices', color='tab:blue', fontsize=13)
    ax2.bar(ind1 + width, avg_data_fed[:3], width, color='tab:blue', label='Non-Linear RealFM')
    ax2.bar(ind2 + width, avg_data_fed[3:5], width, color='tab:blue', hatch='//', label='Linear RealFM')
    ax2.bar(ind3 + width, avg_data_fed[-1], width, color='tab:blue', hatch='o', label='No Mechanism')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.xticks(np.arange(num_baselines) + width / 2, x)
    plt.legend()
    # plt.show()
    plt.savefig('realfm-server-u-8.png', dpi=200)
    '''

    # CLIENT BAR CHART
    # do the same but plot device accuracy

    '''
    x = ['Non-Linear RealFM', 'Linear RealFM', 'Local Training']
    nw = 8
    num_baselines = len(x)
    trials = 3
    avg_acc_fed = np.empty(num_baselines)
    costs = np.array([0.001234, 0.000311725, 0.001234])
    width = 0.3
    matplotlib.rc('ytick', labelcolor='tab:blue')

    fed_acc = 0
    fed_acc2 = 0
    local_acc = 0
    for i in range(num_baselines):

        file = 'output/Cifar10/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 100, nw, datatype='fed-epoch-acc-top1.log')
        fed_acc += optimal_data[-1, 0]
        optimal_data = unpack_data(file, 100, nw, datatype='local-epoch-acc-top1.log')
        local_acc += np.average(optimal_data[-1, :])

        file = 'output/Cifar10/realfm-linear-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 100, nw, datatype='fed-epoch-acc-top1.log')
        fed_acc2 += optimal_data[-1, 0]
        optimal_data = unpack_data(file, 100, nw, datatype='local-epoch-acc-top1.log')
        local_acc += np.average(optimal_data[-1, :])

    # determine average accs
    avg_acc_fed[0] = fed_acc / 3
    avg_acc_fed[1] = fed_acc2 / 3
    avg_acc_fed[2] = local_acc / 6

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # create axis labels
    ax1.set_xlabel('Training Methods', fontsize=13)
    ax1.set_ylabel('Average Device Accuracy', color='tab:red', fontsize=13)
    ax1.set_ylim([0.5, 0.75])
    ax1.bar(0, avg_acc_fed[0], width, color='tab:red')
    ax1.bar(1, avg_acc_fed[1], width, color='tab:red', hatch='//')
    ax1.bar(2, avg_acc_fed[2], width, color='tab:red', hatch='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Adding Twin Axes
    ax2 = ax1.twinx()
    ax2.bar(0 + width, costs[0], width, color='tab:blue')
    ax2.bar(1 + width, costs[1], width, color='tab:blue', hatch='//')
    ax2.bar(2 + width, costs[2], width, color='tab:blue', hatch='o')
    ax2.set_yscale("log")
    ax2.set_ylabel('Maximum Allowable Marginal Cost', color='tab:blue', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.xticks(np.arange(num_baselines) + width / 2, x)
    # plt.show()
    plt.savefig('realfm-device-comparison-8.png', dpi=200)
    '''

    # scatter plot idea
    '''
    plt.figure()

    # plot our federated mechanism (all with same uniform costs)
    for i in range(15):
        avg_data = np.average(local_b[i, :])
        total_fed = np.sum(fed_b[i, :])
        avg_data_fed = np.average(fed_b[i, :])

        # local baseline
        acc_local = accuracy(avg_data, a_opt, k)
        proj_utility_local = accuracy_utility(acc_local, 1, 2)

        # our mechanism baseline
        acc_fed = accuracy(total_fed, a_opt, k)
        proj_utility_fed = accuracy_utility(acc_fed, 1, 2)

        if i // 3 == 0:
            col = 'b'
            lab = 'RealFM (Uniform)'
            mark = 'o'
        elif i // 3 == 1:
            col = 'k'
            lab = 'RealFM (Non-Uniform Payoff and Cost)'
            mark = 'P'
        elif i // 3 == 2:
            col = 'g'
            lab = 'RealFM (Non-Uniform Cost)'
            mark = '*'
        elif i // 3 == 3:
            col = 'tab:purple'
            lab = 'Linear RealFM (Uniform)'
            mark = 'o'
        else:
            col = 'tab:orange'
            lab = 'Linear RealFM (Non-Uniform Cost)'
            mark = '*'

        # plot baselines
        plt.scatter(avg_data, proj_utility_local, color='r', marker='x', label='Local Training' if (i+1) == 3 else "",
                    s=100)
        plt.scatter(avg_data_fed, proj_utility_fed, color=col, marker=mark, label=lab if (i+1) % 3 == 0 else "", s=100)

    plt.ylabel('Server Utility', fontsize=13)
    plt.xlabel('Average Gradient Contributions Across Devices', fontsize=13)
    plt.grid(which="both", alpha=0.25)
    plt.legend(loc='lower right')
    # plt.xlim([5000, 13000])
    # plt.ylim([5, 33])
    plt.show()
    # plt.savefig('realfm-server-u.png', dpi=200)
    '''