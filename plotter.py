import os
import matplotlib.pyplot as plt
import numpy as np
from utils.equilibrium import accuracy_utility, accuracy
from utils.equilibrium import optimal_data_fed


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


class RealFMPlotter:
    def __init__(self, dataset, num_workers, colors, non_iid, dirichlet_value, trials=3):
        self.dataset = dataset
        self.num_workers = num_workers
        self.colors = colors
        self.trials = trials
        self.non_iid = non_iid
        self.dirichlet_value = dirichlet_value

        if dataset == 'mnist':
            if self.non_iid:
                self.file_start = 'output/MNIST/' + str(self.num_workers) + 'dev/noniid/D-' \
                                  + str(self.dirichlet_value) + '/realfm-'
                self.file_end = '-mnist-' + str(self.num_workers) + 'devices-noniid'
            else:
                self.file_start = 'output/MNIST/' + str(self.num_workers) + 'dev/iid/' + 'realfm-'
                self.file_end = '-mnist-' + str(self.num_workers) + 'devices'
            self.epochs = 50
            self.a_opt = 0.995
            self.k = 0.25
        elif dataset == 'cifar10':
            if self.non_iid:
                self.file_start = 'output/Cifar10/' + str(self.num_workers) + 'dev/noniid/D-' \
                                  + str(self.dirichlet_value) + '/realfm-'
                self.file_end = '-cifar10-' + str(self.num_workers) + 'devices-noniid'
            else:
                self.file_start = 'output/Cifar10/' + str(self.num_workers) + 'dev/iid/' + 'realfm-'
                self.file_end = '-cifar10-' + str(self.num_workers) + 'devices'
            self.epochs = 100
            self.a_opt = 0.9
            self.k = 18
        else:
            print('Error')
            exit()

        self.iters = np.arange(1, self.epochs + 1)
        self.x = ['RealFM (Non-Uniform Payoff and Cost)', 'RealFM (Non-Uniform Cost)', 'RealFM (Uniform)',
                  'Linear RealFM (Non-Uniform Cost)', 'Linear RealFM (Uniform)', 'Local Training']
        self.experiments = ['linear-nonuniformC', 'linear-uniform', 'nonuniformPC', 'nonuniformC', 'uniform']

        # get data
        self.get_contribution_data()

    def get_test_accuracy(self, exp):
        fed_accs = []
        local_accs = []
        # each_dev_local_a = []

        for trial in range(1, self.trials + 1):
            file = self.file_start + exp + '-run' + str(trial) + self.file_end
            fed_test_acc = unpack_data(file, self.epochs, self.num_workers, datatype='fed-epoch-acc-top1.log')
            local_test_acc = unpack_data(file, self.epochs, self.num_workers, datatype='local-epoch-acc-top1.log')
            fed_accs.append(fed_test_acc[:, 0])
            local_accs.append(np.mean(local_test_acc, axis=1))
            # each_dev_local_a.append(local_test_acc[-1, :])

        fed_accs = np.stack(fed_accs, axis=0)
        local_accs = np.stack(local_accs, axis=0)
        return fed_accs, local_accs

    def get_contribution_data(self):

        self.local_b = np.ones((15, self.num_workers)) * np.nan
        self.fed_b = np.ones((15, self.num_workers)) * np.nan
        self.payoff = np.ones((15, self.num_workers)) * np.nan
        self.mc = np.ones((15, self.num_workers)) * np.nan

        for trial in range(1, 4):
            file = self.file_start + self.experiments[0] + '-run' + str(trial) + self.file_end
            optimal_data = unpack_data(file, 5, self.num_workers, datatype='update-contribution.log')
            self.mc[trial - 1, :] = optimal_data[0, :]
            self.payoff[trial - 1, :] = optimal_data[1, :]
            self.local_b[trial - 1, :] = optimal_data[3, :]
            self.fed_b[trial - 1, :] = optimal_data[4, :]

            file = self.file_start + self.experiments[1] + '-run' + str(trial) + self.file_end
            optimal_data = unpack_data(file, 5, self.num_workers, datatype='update-contribution.log')
            self.mc[trial + 2, :] = optimal_data[0, :]
            self.payoff[trial + 2, :] = optimal_data[1, :]
            self.local_b[trial + 2, :] = optimal_data[3, :]
            self.fed_b[trial + 2, :] = optimal_data[4, :]

            file = self.file_start + self.experiments[2] + '-run' + str(trial) + self.file_end
            optimal_data = unpack_data(file, 5, self.num_workers, datatype='update-contribution.log')
            self.mc[trial + 5, :] = optimal_data[0, :]
            self.payoff[trial + 5, :] = optimal_data[1, :]
            self.local_b[trial + 5, :] = optimal_data[3, :]
            self.fed_b[trial + 5, :] = optimal_data[4, :]

            file = self.file_start + self.experiments[3] + '-run' + str(trial) + self.file_end
            optimal_data = unpack_data(file, 5, self.num_workers, datatype='update-contribution.log')
            self.mc[trial + 8, :] = optimal_data[0, :]
            self.payoff[trial + 8, :] = optimal_data[1, :]
            self.local_b[trial + 8, :] = optimal_data[3, :]
            self.fed_b[trial + 8, :] = optimal_data[4, :]

            file = self.file_start + self.experiments[4] + '-run' + str(trial) + self.file_end
            optimal_data = unpack_data(file, 5, self.num_workers, datatype='update-contribution.log')
            self.mc[trial + 11, :] = optimal_data[0, :]
            self.payoff[trial + 11, :] = optimal_data[1, :]
            self.local_b[trial + 11, :] = optimal_data[3, :]
            self.fed_b[trial + 11, :] = optimal_data[4, :]

    def test_accuracy_plot(self, save_figure):

        # get test accuracies
        fed_accs, local_accs = self.get_test_accuracy('uniform')

        # compute error bars over all three runs
        y_mean, y_min, y_max = generate_confidence_interval(fed_accs)
        y_mean_local, y_min_local, y_max_local = generate_confidence_interval(local_accs)

        plt.figure(figsize=(8, 6))

        # plot federated results
        plt.plot(self.iters, y_mean, label='Federated Training', color='b')
        plt.fill_between(self.iters, y_min, y_max, alpha=0.2, color='b')

        # plot average of local results
        plt.plot(self.iters, y_mean_local, label='Average Local Training', color='r')
        plt.fill_between(self.iters, y_min_local, y_max_local, alpha=0.2, color='r')

        # title = 'Federated Training vs. Average Local Training for CIFAR-10'
        # plt.title(title)
        # plt.ylim([0.225, 0.48])

        plt.legend(loc='lower right')
        plt.ylabel('Test Accuracy', fontsize=13)
        plt.xlabel('Epochs', fontsize=13)
        plt.xlim([1, self.epochs])
        plt.xscale("log")
        plt.grid(which="both", alpha=0.25)

        if self.dataset == 'mnist':
            plt.ylim([0.1, 1])
        else:
            plt.ylim([0.075, 0.76])

        # plt.tight_layout()
        if save_figure:
            savefilename = 'accuracy-comp-' + self.file_end + '.png'
            plt.savefig(savefilename, dpi=200)
        else:
            plt.show()

    def contribution_bar_chart(self, save_figure):

        for i, exp in enumerate(self.experiments):

            # plot data
            local_b_mean = np.nanmean(self.local_b[i*3:(i+1)*3, :], axis=0)
            fed_b_mean = np.nanmean(self.fed_b[i*3:(i+1)*3, :], axis=0)

            # bar plot
            added_b = fed_b_mean - local_b_mean
            devices = np.arange(1, self.num_workers + 1)
            plt.figure()
            plt.bar(devices, local_b_mean, label='Local Optimal Contributions $(g_i^o)$')
            bar = plt.bar(devices, added_b, bottom=local_b_mean, label='Incentivized Contributions $(g_i^* - g_i^o)$')
            plt.xlabel('Devices', fontsize=13)
            plt.ylabel('Average Gradient Contribution ($g_i$)', fontsize=13)
            plt.xticks(devices)
            plt.legend(loc='lower left', fontsize=10)
            # plt.ylim([0, 5500])

            for i, rect in enumerate(bar):
                val = 100 * (added_b[i] / local_b_mean[i])
                height = rect.get_height() + local_b_mean[i]
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'+{val:.0f}%', ha='center', va='bottom',
                         fontsize=6)

            if save_figure:
                savefilename = str(self.num_workers) + 'device-bar-mean-' + exp + '-' + self.dataset + '.png'
                plt.savefig(savefilename, dpi=200)
            else:
                plt.show()

    def server_utility_comparison(self, save_figure, trials=3):

        x = ['U-LP', 'U-NLP', 'NU-C', 'U-LP', 'NU-PC', 'NU-C', 'U-NLP']

        # local baseline (expected)
        expected_local_b = None
        if self.num_workers == 16:
            expected_local_b = 3000
        elif self.num_workers == 8:
            expected_local_b = 5500

        num_baselines = len(x)
        avg_data_fed = np.empty(num_baselines)
        avg_utility_fed = np.empty(num_baselines)
        avg_data_fed[:2] = expected_local_b
        a_local = accuracy(expected_local_b, self.a_opt, self.k)
        avg_utility_fed[0] = a_local
        avg_utility_fed[1] = accuracy_utility(a_local, 1, 2)

        local_ind = np.arange(2)
        linear_ind = np.arange(2, 4)
        nonlinear_ind = np.arange(4, 7)

        for i in range(num_baselines - 2):
            all_data_avg = np.average(np.sum(self.fed_b[trials * i:(trials * (i + 1)), :], axis=1))
            total_fed = np.average(self.fed_b[trials * i:(trials * (i + 1)), :], axis=1)
            avg_data_fed[i + 2] = np.average(total_fed)

            # mechanism baseline
            acc_fed = accuracy(all_data_avg, self.a_opt, self.k)
            if i < 2:
                # linear utility
                avg_utility_fed[i + 2] = acc_fed
            else:
                # non-linear utility
                avg_utility_fed[i + 2] = accuracy_utility(acc_fed, 1, 2)

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

        # dataset specific plot parameters
        max_u = np.max(avg_utility_fed)
        if self.dataset == 'mnist':
            max_u *= 4.7
            ax1.set_yscale('log')
            val = np.log(max_u)
            avg_acc_util_plt = np.log(avg_utility_fed)
            offsets = [0.04, 0.06, 0.02]
        else:
            max_u = np.ceil(max_u*1.2)
            val = max_u
            avg_acc_util_plt = avg_utility_fed
            offsets = [0.05, 0.03, 0.02]

        ax1.set_ylim([0, max_u])
        local_h = (np.max(avg_acc_util_plt[:2]) / val) + offsets[0]
        linear_h = (np.max(avg_acc_util_plt[2:4]) / val) + offsets[1]
        nonlinear_h = (np.max(avg_acc_util_plt[4:]) / val) + offsets[2]

        ax1.annotate('Non-Linear RealFM', xy=(0.781, nonlinear_h), xytext=(0.781, nonlinear_h + 0.05),
                     xycoords='axes fraction',
                     fontsize=18, ha='center', va='bottom', weight='bold', color='tab:blue',
                     bbox=dict(boxstyle='square', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=6.5, lengthB=1', lw=2.0, color='k'))

        ax1.annotate('Linear RealFM', xy=(0.43, linear_h), xytext=(0.43, linear_h + 0.05), xycoords='axes fraction',
                     fontsize=18, ha='center', va='bottom', weight='bold', color='tab:green',
                     bbox=dict(boxstyle='square', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=4.0, lengthB=1', lw=2.0, color='k'))

        ax1.annotate('Linear & Non-Linear\n Local Training', xy=(0.1425, local_h), xytext=(0.1425, local_h + 0.05),
                     xycoords='axes fraction',
                     fontsize=18, ha='center', va='bottom', weight='bold', color='tab:red',
                     bbox=dict(boxstyle='square', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=4.25, lengthB=1', lw=2.0, color='k'))

        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.tick_params(axis='y', which='major', labelsize=16)
        plt.tight_layout()

        if save_figure:
            title = 'realfm-server-utility-' + str(self.num_workers) + 'devices-' + self.dataset + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

    def device_contribution_comparison(self, save_figure, trials=3):

        x = ['U-LP', 'U-NLP', 'NU-C', 'U-LP', 'NU-PC', 'NU-C', 'U-NLP']

        # local baseline (expected)
        expected_local_b = None
        if self.num_workers == 16:
            expected_local_b = 3000
        elif self.num_workers == 8:
            expected_local_b = 5500

        num_baselines = len(x)
        avg_data_fed = np.empty(num_baselines)
        avg_data_fed[:2] = expected_local_b

        local_ind = np.arange(2)
        linear_ind = np.arange(2, 4)
        nonlinear_ind = np.arange(4, 7)

        for i in range(num_baselines - 2):
            total_fed = np.average(self.fed_b[trials * i:(trials * (i + 1)), :], axis=1)
            avg_data_fed[i + 2] = np.average(total_fed)

        width = 0.45
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.set_ylabel('Average Device Data Contribution', fontsize=18, weight='bold')
        ax2.bar(local_ind, avg_data_fed[:2], width, color='tab:red')
        ax2.bar(linear_ind, avg_data_fed[2:4], width, color='tab:green')
        ax2.bar(nonlinear_ind, avg_data_fed[4:], width, color='tab:blue')
        ax2.grid(axis='y', alpha=0.25)
        ax2.set_yscale('log')

        plt.xticks(np.arange(num_baselines), x, weight='bold', fontsize=15)

        if self.dataset == 'mnist':
            offsets = [0.01, 0.05, 0.01]
        else:
            offsets = [0.01, 0.02, 0.01]

        max_u = np.max(avg_data_fed) * 1.15
        val = np.log(max_u)
        log_avg_data_fed = np.log(avg_data_fed)
        ax2.set_ylim([0, max_u])

        local_h = (np.max(log_avg_data_fed[:2]) / val) + offsets[0]
        linear_h = (np.max(log_avg_data_fed[2:4]) / val) + offsets[1]
        nonlinear_h = (np.max(log_avg_data_fed[4:]) / val) + offsets[2]

        ax2.annotate('Non-Linear RealFM', xy=(0.781, nonlinear_h), xytext=(0.781, nonlinear_h + 0.05),
                     xycoords='axes fraction',
                     fontsize=18, ha='center', va='bottom', weight='bold', color='tab:blue',
                     bbox=dict(boxstyle='square', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=6.5, lengthB=1', lw=2.0, color='k'))

        ax2.annotate('Linear RealFM', xy=(0.4275, linear_h), xytext=(0.4275, linear_h + 0.05), xycoords='axes fraction',
                     fontsize=18, ha='center', va='bottom', weight='bold', color='tab:green',
                     bbox=dict(boxstyle='square', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=4.0, lengthB=0.75', lw=2.0, color='k'))

        ax2.annotate('Linear & Non-Linear\n Local Training', xy=(0.14, local_h), xytext=(0.14, local_h + 0.05),
                     xycoords='axes fraction',
                     fontsize=18, ha='center', va='bottom', weight='bold', color='tab:red',
                     bbox=dict(boxstyle='square', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=4.25, lengthB=1', lw=2.0, color='k'))

        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(axis='y', which='major', labelsize=16)
        plt.tight_layout()

        if save_figure:
            title = 'realfm-server-data-produced-' + str(self.num_workers) + 'devices-' + self.dataset + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

    def device_utility_comparison(self, save_figure):

        # x = ['Uniform\n (L Payoff)', 'Uniform\n (NL Payoff)', 'Uniform', 'Uniform']
        x = ['U-LP', 'U-NLP', 'U', 'U']
        num_baselines = len(x)
        avg_acc_util = np.zeros(num_baselines)
        width = 0.5

        fed_accs_linear, local_accs_linear = self.get_test_accuracy('linear-uniform')
        fed_accs, local_accs = self.get_test_accuracy('uniform')

        avg_acc_util[0] = np.average(local_accs_linear[:, -1])
        avg_acc_util[1] = accuracy_utility(np.average(local_accs[:, -1]), 1, 2)
        avg_acc_util[2] = np.average(fed_accs_linear[:, -1])
        avg_acc_util[3] = accuracy_utility(np.average(fed_accs[:, -1]), 1, 2)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # dataset specific plot parameters
        max_u = np.max(avg_acc_util)
        if self.dataset == 'mnist':
            max_u *= 10
            ax1.set_yscale('log')
            val = np.log(max_u)
            avg_acc_util_plt = np.log(avg_acc_util)
            ax1.set_ylim([0.5, 0.75])
            offsets = [0.045, 0.06, 0.02]
        else:
            max_u = np.ceil(max_u * 1.15)
            val = max_u
            avg_acc_util_plt = avg_acc_util
            offsets = [0.02] * 3

        local_h = (np.max(avg_acc_util_plt[:2]) / val) + offsets[0]
        linear_h = (avg_acc_util_plt[2] / val) + offsets[1]
        nonlinear_h = (avg_acc_util_plt[3] / val) + offsets[2]
        ax1.set_ylim([0, max_u])

        # create axis labels
        ax1.set_ylabel('Average Device Utility via Accuracy', fontsize=18, weight='bold')
        ax1.bar(0, avg_acc_util[0], width, color='tab:red')
        ax1.bar(1, avg_acc_util[1], width, color='tab:red')
        ax1.bar(2, avg_acc_util[2], width, color='tab:green')
        ax1.bar(3, avg_acc_util[3], width, color='tab:blue')
        ax1.grid(axis='y', alpha=0.25)
        plt.xticks(np.arange(num_baselines), x, weight='bold', fontsize=18)

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

        plt.tight_layout()
        ax1.tick_params(axis='y', which='major', labelsize=16)

        if save_figure:
            title = 'realfm-device-utility-' + str(self.num_workers) + 'devices-' + self.dataset + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

if __name__ == '__main__':
    clr = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']
    num_w = 16
    ds = 'mnist'
    non_iid = True
    dirichlet_value = 0.6

    plotter = RealFMPlotter(ds, num_w, clr, non_iid, dirichlet_value)
    # plotter.contribution_bar_chart(False)
    plotter.device_contribution_comparison(False)
    # plotter.server_utility_comparison(False)
    # plotter.device_utility_comparison(False)
    # plotter.test_accuracy_plot(False)
