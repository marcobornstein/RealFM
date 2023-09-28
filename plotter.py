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
    nw = 16
    epochs = 50
    fed_accs = []
    local_accs = []
    iters = np.arange(1, epochs + 1)
    each_dev_local_a = []

    # strictly accuracy plots
    '''
    for trial in range(1, 4):
        # file = 'output/Cifar10/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        file = 'output/MNIST/realfm-uniform-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
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
    savefilename = str(nw) + 'device-mnist.png'
    plt.savefig(savefilename, dpi=200)
    '''

    # =======================================================
    # plot data contribution plot and bar chart

    # '''
    nw = 8
    # a_opt = 0.9
    # k = 18
    a_opt = 0.995
    k = 0.25
    local_b = np.ones((15, nw)) * np.nan
    fed_b = np.ones((15, nw)) * np.nan
    payoff = np.ones((15, nw)) * np.nan
    mc = np.ones((15, nw)) * np.nan
    x = ['RealFM (Non-Uniform Payoff and Cost)', 'RealFM (Non-Uniform Cost)', 'RealFM (Uniform)',
         'Linear RealFM (Non-Uniform Cost)', 'Linear RealFM (Uniform)', 'Local Training']

    for trial in range(1, 4):

        # file = 'output/Cifar10/realfm-linear-nonuniformC-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        file = 'output/MNIST/realfm-linear-nonuniformC-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial - 1, :] = optimal_data[0, :]
        payoff[trial - 1, :] = optimal_data[1, :]
        local_b[trial - 1, :] = optimal_data[2, :]
        fed_b[trial - 1, :] = optimal_data[3, :]

        # file = 'output/Cifar10/realfm-linear-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        file = 'output/MNIST/realfm-linear-uniform-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 2, :] = optimal_data[0, :]
        payoff[trial + 2, :] = optimal_data[1, :]
        local_b[trial + 2, :] = optimal_data[2, :]
        fed_b[trial + 2, :] = optimal_data[3, :]

        # file = 'output/Cifar10/realfm-nonuniformPC-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        file = 'output/MNIST/realfm-nonuniformPC-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 5, :] = optimal_data[0, :]
        payoff[trial + 5, :] = optimal_data[1, :]
        local_b[trial + 5, :] = optimal_data[2, :]
        fed_b[trial + 5, :] = optimal_data[3, :]

        # file = 'output/Cifar10/realfm-nonuniformC-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        file = 'output/MNIST/realfm-nonuniformC-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 8, :] = optimal_data[0, :]
        payoff[trial + 8, :] = optimal_data[1, :]
        local_b[trial + 8, :] = optimal_data[2, :]
        fed_b[trial + 8, :] = optimal_data[3, :]

        # file = 'output/Cifar10/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        file = 'output/MNIST/realfm-uniform-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 4, nw, datatype='update-contribution.log')
        mc[trial + 11, :] = optimal_data[0, :]
        payoff[trial + 11, :] = optimal_data[1, :]
        local_b[trial + 11, :] = optimal_data[2, :]
        fed_b[trial + 11, :] = optimal_data[3, :]

    '''
    # local_a = np.mean(np.stack(each_dev_local_a, axis=0), axis=0)
    local_b_mean = np.nanmean(local_b[:3], axis=0)
    fed_b_mean = np.nanmean(fed_b[:3], axis=0)

    # bar plot
    added_b = fed_b_mean - local_b_mean
    devices = np.arange(1, nw+1)
    plt.figure()
    plt.bar(devices, local_b_mean, label='Local Optimal Contributions $(g_i^o)$')
    bar = plt.bar(devices, added_b, bottom=local_b_mean, label='Incentivized Contributions $(g_i^* - g_i^o)$')
    plt.xlabel('Devices', fontsize=13)
    plt.ylabel('Average Gradient Contribution ($g_i$)', fontsize=13)
    plt.xticks(devices)
    plt.legend(loc='lower left', fontsize=10)
    # plt.ylim([0, 5500])

    for i, rect in enumerate(bar):
        val = 100*(added_b[i] / local_b_mean[i])
        height = rect.get_height() + local_b_mean[i]
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'+{val:.0f}%', ha='center', va='bottom', fontsize=6)

    savefilename2 = str(nw) + 'device-bar-mean-linear-nonuniformC-cifar.png'
    plt.savefig(savefilename2, dpi=200)
    # plt.show()
    '''

    # bar chart idea: plot for each test type (including local baseline)
    # SERVER BAR CHART
    # both 1) average gradient contributions across devices and 2) server utility

    # '''

    # x = ['Uniform\n (L Payoff)', 'Uniform\n (NL Payoff)', 'N-Uniform\n Cost', 'Uniform',
    #      'N-Uniform  \n Payoff & Cost', 'N-Uniform\n Cost', 'Uniform']

    x = ['U-LP', 'U-NLP', 'NU-C', 'U-LP', 'NU-PC', 'NU-C', 'U-NLP']

    # x = ['\n'.join(wrap(l, 15)) for l in x]

    # local baseline (expected)
    if nw == 16:
        expected_local_b = 3000
    elif nw == 8:
        expected_local_b = 5500

    num_baselines = len(x)
    trials = 3
    avg_data_fed = np.empty(num_baselines)
    avg_utility_fed = np.empty(num_baselines)
    avg_data_fed[:2] = expected_local_b
    a_local = accuracy(expected_local_b, a_opt, k)
    avg_utility_fed[0] = a_local
    avg_utility_fed[1] = accuracy_utility(a_local, 1, 2)

    local_ind = np.arange(2)
    linear_ind = np.arange(2, 4)
    nonlinear_ind = np.arange(4, 7)

    for i in range(num_baselines-2):
        all_data_avg = np.average(np.sum(fed_b[trials*i:(trials*(i+1)), :], axis=1))
        total_fed = np.average(fed_b[trials*i:(trials*(i+1)), :], axis=1)
        avg_data_fed[i+2] = np.average(total_fed)

        # mechanism baseline
        acc_fed = accuracy(all_data_avg, a_opt, k)
        if i < 2:
            # linear utility
            avg_utility_fed[i+2] = acc_fed
        else:
            # non-linear utility
            avg_utility_fed[i+2] = accuracy_utility(acc_fed, 1, 2)

    width = 0.45
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # create axis labels
    # ax1.set_xlabel('Training Methods', fontsize=13)
    ax1.set_ylabel('Server Utility', fontsize=18, weight='bold')
    ax1.bar(local_ind, avg_utility_fed[:2], width, color='tab:red')
    ax1.bar(linear_ind, avg_utility_fed[2:4], width, color='tab:green')
    ax1.bar(nonlinear_ind, avg_utility_fed[4:], width, color='tab:blue')
    plt.xticks(np.arange(num_baselines), x, weight='bold', fontsize=15)
    ax1.grid(axis='y', alpha=0.25)

    max_u = np.max(avg_utility_fed)*4.7
    # max_u = np.max(avg_utility_fed) * 1.15

    # for mnist
    ax1.set_yscale('log')
    ax1.set_ylim([0, max_u])

    # nonlinear_h = (np.max(avg_utility_fed[4:]) / max_u) + 0.01
    # linear_h = (np.max(avg_utility_fed[2:4]) / max_u) + 0.02
    # local_h = (np.max(avg_utility_fed[:2]) / max_u) + 0.02

    nonlinear_h = 0.87
    linear_h = 0.07
    local_h = 0.53

    ax1.annotate('Non-Linear RealFM', xy=(0.781, nonlinear_h), xytext=(0.781, nonlinear_h+0.05), xycoords='axes fraction',
                fontsize=18, ha='center', va='bottom', weight='bold', color='tab:blue',
                bbox=dict(boxstyle='square', fc='white', color='k'),
                arrowprops=dict(arrowstyle='-[, widthB=6.5, lengthB=1', lw=2.0, color='k'))

    ax1.annotate('Linear RealFM', xy=(0.43, linear_h), xytext=(0.43, linear_h+0.05), xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:green',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=4.0, lengthB=1', lw=2.0, color='k'))

    ax1.annotate('Linear & Non-Linear\n Local Training', xy=(0.1425, local_h), xytext=(0.1425, local_h+0.05), xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:red',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=4.25, lengthB=1', lw=2.0, color='k'))

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.tick_params(axis='y', which='major', labelsize=16)


    plt.tight_layout()
    # plt.show()
    title = 'realfm-server-utility-' + str(nw) + 'devices-mnist.png'
    plt.savefig(title, dpi=200)

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.set_ylabel('Average Device Data Contribution', fontsize=18, weight='bold')
    ax2.bar(local_ind, avg_data_fed[:2], width, color='tab:red')
    ax2.bar(linear_ind, avg_data_fed[2:4], width, color='tab:green')
    ax2.bar(nonlinear_ind, avg_data_fed[4:], width, color='tab:blue')
    ax2.grid(axis='y', alpha=0.25)

    plt.xticks(np.arange(num_baselines), x, weight='bold', fontsize=15)

    max_u = np.max(avg_data_fed)*1.15
    ax2.set_ylim([0, max_u])

    nonlinear_h = (np.max(avg_data_fed[4:]) / max_u) + 0.01
    linear_h = (np.max(avg_data_fed[2:4]) / max_u) + 0.02
    local_h = (np.max(avg_data_fed[:2]) / max_u) + 0.02

    ax2.annotate('Non-Linear RealFM', xy=(0.781, nonlinear_h), xytext=(0.781, nonlinear_h + 0.05),
                 xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:blue',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=6.5, lengthB=1', lw=2.0, color='k'))

    ax2.annotate('Linear RealFM', xy=(0.4275, linear_h), xytext=(0.4275, linear_h + 0.05), xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:green',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=4.0, lengthB=1', lw=2.0, color='k'))

    ax2.annotate('Linear & Non-Linear\n Local Training', xy=(0.14, local_h), xytext=(0.14, local_h + 0.05),
                 xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:red',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=4.25, lengthB=1', lw=2.0, color='k'))

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis='y', which='major', labelsize=16)

    plt.tight_layout()
    # plt.show()
    title = 'realfm-server-data-produced-' + str(nw) + 'devices-mnist.png'
    plt.savefig(title, dpi=200)
    # '''

    # CLIENT BAR CHART
    # do the same but plot device accuracy and device utility

    '''
    # x = ['Uniform\n (L Payoff)', 'Uniform\n (NL Payoff)', 'Uniform', 'Uniform']
    x = ['U-LP', 'U-NLP', 'U', 'U']
    num_baselines = len(x)
    trials = 3
    avg_acc = np.zeros(num_baselines)
    avg_acc_util = np.zeros(num_baselines)
    width = 0.5
    epochs = 100

    for trial in range(1, 4):

        file = 'output/Cifar10/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        # file = 'output/MNIST/realfm-uniform-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, epochs, nw, datatype='fed-epoch-acc-top1.log')
        avg_acc[3] += optimal_data[-1, 0]
        optimal_data = unpack_data(file, epochs, nw, datatype='local-epoch-acc-top1.log')
        avg_acc[1] += np.average(optimal_data[-1, :])

        file = 'output/Cifar10/realfm-linear-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        # file = 'output/MNIST/realfm-linear-uniform-run' + str(trial) + '-mnist-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, epochs, nw, datatype='fed-epoch-acc-top1.log')
        avg_acc[2] += optimal_data[-1, 0]
        optimal_data = unpack_data(file, epochs, nw, datatype='local-epoch-acc-top1.log')
        avg_acc[0] += np.average(optimal_data[-1, :])

    # determine average accs
    avg_acc = avg_acc / 3
    avg_acc_util[0] = avg_acc[0]
    avg_acc_util[1] = accuracy_utility(avg_acc[1], 1, 2)
    avg_acc_util[2] = avg_acc[2]
    avg_acc_util[3] = accuracy_utility(avg_acc[3], 1, 2)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # create axis labels
    # ax1.set_xlabel('Training Methods', fontsize=13)
    ax1.set_ylabel('Average Device Utility via Accuracy', fontsize=18, weight='bold')
    # ax1.set_yscale('log')
    # ax1.set_ylim([0.5, 0.75])
    ax1.bar(0, avg_acc_util[0], width, color='tab:red')
    ax1.bar(1, avg_acc_util[1], width, color='tab:red')
    ax1.bar(2, avg_acc_util[2], width, color='tab:green')
    ax1.bar(3, avg_acc_util[3], width, color='tab:blue')
    ax1.grid(axis='y', alpha=0.25)
    plt.xticks(np.arange(num_baselines), x, weight='bold', fontsize=18)

    # max_u = 350000
    max_u = np.max(avg_acc_util) * 1.2
    ax1.set_ylim([0, max_u])

    local_h = (np.max(avg_acc_util[:2]) / max_u) + 0.02
    linear_h = (avg_acc_util[2] / max_u) + 0.02
    nonlinear_h = (avg_acc_util[3] / max_u) + 0.02
    # local_h = 0.68
    # linear_h = 0.06
    # nonlinear_h = 0.84

    ax1.annotate('Non-Linear RealFM', xy=(0.89, nonlinear_h), xytext=(0.89, nonlinear_h + 0.05),
                 xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:blue',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=3, lengthB=1', lw=2.0, color='k'))

    ax1.annotate('Linear RealFM', xy=(0.63, linear_h), xytext=(0.63, linear_h + 0.05), xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:green',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=3, lengthB=1', lw=2.0, color='k'))

    ax1.annotate('Linear & Non-Linear\n Local Training', xy=(0.235, local_h), xytext=(0.235, local_h + 0.05),
                 xycoords='axes fraction',
                 fontsize=18, ha='center', va='bottom', weight='bold', color='tab:red',
                 bbox=dict(boxstyle='square', fc='white', color='k'),
                 arrowprops=dict(arrowstyle='-[, widthB=7.25, lengthB=1', lw=2.0, color='k'))
    # ax1.set_ylim([0, 14.3])
    plt.tight_layout()
    ax1.tick_params(axis='y', which='major', labelsize=16)

    # plt.show()
    title = 'realfm-device-utility-' + str(nw) + 'device.png'
    plt.savefig(title, dpi=200)
    '''
