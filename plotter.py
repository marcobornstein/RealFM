import os
import matplotlib.pyplot as plt
import numpy as np
from utils.equilibrium import accuracy_utility, accuracy


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
        self.experiments = ['linear-uniform', 'linear-nonuniformC', 'linear-nonuniformPC', 'uniform',
                            'nonuniformC', 'nonuniformPC']

        # get data
        self.get_data()

    def get_data(self):

        self.local_b = np.empty((self.trials*len(self.experiments), self.num_workers))
        self.fed_b = np.empty((self.trials*len(self.experiments), self.num_workers))
        self.payoff = np.empty((self.trials*len(self.experiments), self.num_workers))
        self.mc = np.empty((self.trials*len(self.experiments), self.num_workers))

        for trial in range(1, self.trials+1):
            for i, exp in enumerate(self.experiments):
                file = self.file_start + exp + '-run' + str(trial) + self.file_end
                optimal_data = unpack_data(file, 5, self.num_workers, datatype='update-contribution.log')
                self.mc[(trial - 1)+(i*self.trials), :] = optimal_data[0, :]
                self.payoff[(trial - 1)+(i*self.trials), :] = optimal_data[1, :]
                self.local_b[(trial - 1)+(i*self.trials), :] = optimal_data[3, :]
                self.fed_b[(trial - 1)+(i*self.trials), :] = optimal_data[4, :]

                if np.sum(optimal_data[3, :]) == 0:
                    print(file)
                    print('Error: Corrupt File')
                    exit()

        num_baselines = len(self.experiments)
        self.avg_data_local = np.empty(num_baselines)
        self.avg_acc_local = np.empty(num_baselines)
        self.avg_utility_local = np.empty(num_baselines)
        self.avg_data_fed = np.empty(num_baselines)
        self.avg_acc_fed = np.empty(num_baselines)
        self.avg_utility_fed = np.empty(num_baselines)

        for i, exp in enumerate(self.experiments):

            # get initial local amount of data given the marginal cost
            all_device_local_data = self.local_b[self.trials * i:(self.trials * (i + 1)), :]
            total_local_data_avg = np.average(np.sum(all_device_local_data, axis=1))
            avg_local_data_per_device = np.average(np.average(all_device_local_data, axis=1))

            # store average local accuracy and data

            self.avg_acc_local[i] = accuracy(avg_local_data_per_device, self.a_opt, self.k)
            self.avg_data_local[i] = avg_local_data_per_device

            '''
            # TODO: Ensure this is correct (maybe make this the case for every value)
            # if self.avg_acc_local[i] <= np.inf:
            if self.avg_acc_local[i] <= 0:
                print('not enough original data')
                _, local_a = self.get_test_accuracy(exp)
                self.avg_acc_local[i] = np.average(local_a[:, -1])
            '''

            # get expected amount of data via federated mechanism
            all_device_fed_data = self.fed_b[self.trials * i:(self.trials * (i + 1)), :]
            total_fed_data_avg = np.average(np.sum(all_device_fed_data, axis=1))
            avg_fed_data_per_device = np.average(np.average(all_device_fed_data, axis=1))

            # if accuracy shaping actually results in less data than local, use just local data since device
            # wouldn't partake in federated mechanism
            p = all_device_fed_data - all_device_local_data
            print(p[:3, :])
            if total_fed_data_avg < total_local_data_avg:
                print('not enough epochs to get fed acc large enough')
                self.avg_acc_fed[i] = self.avg_acc_local[i]
                self.avg_data_fed[i] = avg_local_data_per_device
            else:
                self.avg_acc_fed[i] = accuracy(total_fed_data_avg, self.a_opt, self.k)
                self.avg_data_fed[i] = avg_fed_data_per_device

            # compute utility
            if exp.find('linear') == 0:
                # linear utility for local and fed
                self.avg_utility_fed[i] = self.avg_acc_fed[i]
                self.avg_utility_local[i] = self.avg_acc_local[i]
            else:
                # non-linear utility for local and fed
                self.avg_utility_fed[i] = accuracy_utility(self.avg_acc_fed[i], 1, 2)
                self.avg_utility_local[i] = accuracy_utility(self.avg_acc_local[i], 1, 2)

    def device_utility_comparison(self, save_figure):

        x = ['Uniform', 'Non-Uniform C', 'Non-Uniform C&P']

        linear_local_ind = np.array([0, 4, 8])
        linear_fed_ind = linear_local_ind + 1
        nonlinear_local_ind = linear_local_ind + 2
        nonlinear_fed_ind = linear_local_ind + 3
        tick_ind = np.array([1.5, 5.5, 9.5])

        width = 0.4
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # create axis labels
        ax1.set_ylabel('Average Device Utility via Accuracy', fontsize=18, weight='bold')
        ax1.bar(linear_local_ind, self.avg_utility_local[:3], width, color='tab:orange', label='Local Linear')
        ax1.bar(linear_fed_ind, self.avg_utility_fed[:3], width, color='tab:red', label='Linear RealFM')
        ax1.bar(nonlinear_local_ind, self.avg_utility_local[3:], width, color='tab:green', label='Local Non-linear')
        ax1.bar(nonlinear_fed_ind, self.avg_utility_fed[3:], width, color='tab:blue', label='Non-linear RealFM')

        if self.dataset == 'mnist':
            plt.ylim([0, 1e5])
        else:
            plt.ylim([0, 1e2])

        plt.xticks(tick_ind, x, weight='bold', fontsize=15)
        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('symlog')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.tick_params(axis='y', which='major', labelsize=16)
        plt.legend(fontsize=13)
        plt.tight_layout()

        if save_figure:
            title = 'realfm-average-device-utility-' + str(self.num_workers) + 'devices-' + self.dataset + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

    def server_utility_comparison(self, save_figure):

        x = ['Uniform', 'Non-Uniform C', 'Non-Uniform C&P']

        linear_fed_ind = np.array([0, 2, 4])
        nonlinear_fed_ind = linear_fed_ind + 1
        tick_ind = np.array([0.5, 2.5, 4.5])

        width = 0.4
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # create axis labels
        ax1.set_ylabel('Server Utility', fontsize=18, weight='bold')
        ax1.bar(linear_fed_ind, self.avg_utility_fed[:3], width, color='tab:red', label='Linear RealFM')
        ax1.bar(nonlinear_fed_ind, self.avg_utility_fed[3:], width, color='tab:blue', label='Non-linear RealFM')

        if self.dataset == 'mnist':
            plt.ylim([0, 1e5])
        else:
            plt.ylim([0, 1e2])

        plt.xticks(tick_ind, x, weight='bold', fontsize=15)
        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('symlog')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.tick_params(axis='y', which='major', labelsize=16)
        plt.legend(fontsize=13)
        plt.tight_layout()

        if save_figure:
            title = 'realfm-server-utility-' + str(self.num_workers) + 'devices-' + self.dataset + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

    def device_contribution_comparison(self, save_figure):

        x = ['Uniform', 'Non-Uniform C', 'Non-Uniform C&P']
        linear_local_ind = np.array([0, 4, 8])
        linear_fed_ind = linear_local_ind + 1
        nonlinear_local_ind = linear_local_ind + 2
        nonlinear_fed_ind = linear_local_ind + 3
        tick_ind = np.array([1.5, 5.5, 9.5])

        '''
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
        '''

        # create axis labels
        width = 0.4
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_ylabel('Average Device Data Contribution', fontsize=18, weight='bold')
        ax1.bar(linear_local_ind, self.avg_data_local[:3], width, color='tab:orange', label='Local Linear')
        ax1.bar(linear_fed_ind, self.avg_data_fed[:3], width, color='tab:red', label='Linear RealFM')
        ax1.bar(nonlinear_local_ind, self.avg_data_local[3:], width, color='tab:green', label='Local Non-linear')
        ax1.bar(nonlinear_fed_ind, self.avg_data_fed[3:], width, color='tab:blue', label='Non-linear RealFM')
        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('log')

        plt.xticks(tick_ind, x, weight='bold', fontsize=15)
        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('symlog')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.tick_params(axis='y', which='major', labelsize=16)
        plt.legend(fontsize=13)

        if self.dataset == 'mnist':
            plt.ylim([0, 2e8])
        else:
            plt.ylim([0, 1e2])

        plt.tight_layout()

        if save_figure:
            title = 'realfm-average-device-contribution-' + str(self.num_workers) + 'devices-' + self.dataset + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

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

    def contribution_bar_chart(self, save_figure):

        for i, exp in enumerate(self.experiments):

            # plot data
            local_b_mean = np.nanmean(self.local_b[i * 3:(i + 1) * 3, :], axis=0)
            fed_b_mean = np.nanmean(self.fed_b[i * 3:(i + 1) * 3, :], axis=0)

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

if __name__ == '__main__':
    clr = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']
    num_w = 8
    ds = 'mnist'
    non_iid = True
    dirichlet_value = 0.3

    plotter = RealFMPlotter(ds, num_w, clr, non_iid, dirichlet_value)
    # plotter.contribution_bar_chart(False)
    plotter.device_contribution_comparison(False)
    # plotter.server_utility_comparison(False)
    # plotter.device_utility_comparison(False)
    # plotter.test_accuracy_plot(False)
