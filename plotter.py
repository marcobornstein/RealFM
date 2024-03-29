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
            self.epochs = 100
            self.a_opt = 0.9975
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
            self.a_opt = 0.95
            self.k = 10
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
        self.avg_local_dev_utility = np.empty(num_baselines)
        self.avg_fed_dev_utility = np.empty(num_baselines)

        for i, exp in enumerate(self.experiments):

            # get initial local amount of data given the marginal cost
            all_device_local_data = self.local_b[self.trials * i:(self.trials * (i + 1)), :]
            all_device_mc = self.mc[self.trials * i:(self.trials * (i + 1)), :]
            total_local_data_avg = np.average(np.sum(all_device_local_data, axis=1))
            avg_local_data_per_device = np.average(np.average(all_device_local_data, axis=1))

            # store average local accuracy and data
            self.avg_acc_local[i] = accuracy(avg_local_data_per_device, self.a_opt, self.k)
            self.avg_data_local[i] = avg_local_data_per_device

            # get expected amount of data via federated mechanism
            all_device_fed_data = self.fed_b[self.trials * i:(self.trials * (i + 1)), :]
            total_fed_data_avg = np.average(np.sum(all_device_fed_data, axis=1))
            avg_fed_data_per_device = np.average(np.average(all_device_fed_data, axis=1))

            self.avg_acc_fed[i] = accuracy(total_fed_data_avg, self.a_opt, self.k)
            self.avg_data_fed[i] = avg_fed_data_per_device

            # compute utility
            if exp.find('linear') == 0:
                # linear utility for local and fed
                self.avg_utility_fed[i] = self.avg_acc_fed[i]
                self.avg_fed_dev_utility[i] = self.avg_utility_fed[i]
                self.avg_utility_local[i] = self.avg_acc_local[i]
                self.avg_local_dev_utility[i] = self.avg_utility_local[i]
            else:
                # non-linear utility for local and fed
                self.avg_utility_fed[i] = accuracy_utility(self.avg_acc_fed[i], 1, 2)
                self.avg_fed_dev_utility[i] = self.avg_utility_fed[i]
                self.avg_utility_local[i] = accuracy_utility(self.avg_acc_local[i], 1, 2)
                self.avg_local_dev_utility[i] = self.avg_utility_local[i]

            # update device utility fed & local
            self.avg_fed_dev_utility[i] -= np.average(np.average(all_device_fed_data * all_device_mc, axis=1))
            self.avg_local_dev_utility[i] = -np.average(np.average(all_device_local_data * all_device_mc, axis=1))

    def server_utility_comparison(self, save_figure):

        x = ['Uniform', 'Non-Uniform C', 'Non-Uniform C&P']

        width = 0.75
        linear_local_ind = np.array([0, 3.5, 7])
        linear_fed_ind = linear_local_ind + width
        nonlinear_local_ind = linear_local_ind + 2*width
        nonlinear_fed_ind = linear_local_ind + 3*width
        tick_ind = np.mean([linear_local_ind, linear_fed_ind, nonlinear_local_ind, nonlinear_fed_ind], axis=0)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # create axis labels
        ax1.set_ylabel('Server Utility', fontsize=20, weight='bold')
        ax1.bar(nonlinear_fed_ind, self.avg_utility_fed[3:], width, color='tab:blue', label='Non-linear RealFM')
        ax1.bar(linear_fed_ind, self.avg_utility_fed[:3], width, color='tab:red', label='Linear RealFM')
        ax1.bar(nonlinear_local_ind, self.avg_utility_local[3:], width, color='tab:green', label='Local Non-linear')
        ax1.bar(linear_local_ind, self.avg_utility_local[:3], width, color='tab:orange', label='Local Linear')

        if self.dataset == 'mnist':
            plt.ylim([0, 2e5])
        else:
            plt.ylim([0, 2e2])

        plt.xticks(tick_ind, x, weight='bold', fontsize=20)
        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('symlog')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.tick_params(axis='y', which='major', labelsize=18)
        # plt.legend(fontsize=13, loc='upper left')
        plt.tight_layout()

        if save_figure:
            title = 'realfm-average-server-utility' + self.file_end
            if self.non_iid:
                title = title + str(self.dirichlet_value) + '.png'
            else:
                title = title + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

    def device_utility_comparison(self, save_figure):

        x = ['Uniform', 'Non-Uniform C', 'Non-Uniform C&P']
        width = 0.5
        linear_fed_ind = np.array([0, 1.5, 3])
        nonlinear_fed_ind = linear_fed_ind + width
        tick_ind = (linear_fed_ind + nonlinear_fed_ind) / 2

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # create axis labels
        ax1.set_ylabel('Device Utility', fontsize=20, weight='bold')
        ax1.bar(nonlinear_fed_ind, self.avg_fed_dev_utility[3:], width, color='tab:blue', label='Non-linear RealFM')
        ax1.bar(linear_fed_ind, self.avg_fed_dev_utility[:3], width, color='tab:red', label='Linear RealFM')

        if self.dataset == 'mnist':
            plt.ylim([0, 2e5])
        else:
            plt.ylim([0, 2e2])

        plt.xticks(tick_ind, x, weight='bold', fontsize=20)
        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('symlog')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.tick_params(axis='y', which='major', labelsize=18)
        # plt.legend(fontsize=13)
        plt.tight_layout()

        if save_figure:
            title = 'realfm-device-utility' + self.file_end
            if self.non_iid:
                title = title + str(self.dirichlet_value) + '.png'
            else:
                title = title + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

    def device_contribution_comparison(self, save_figure):

        x = ['Uniform', 'Non-Uniform C', 'Non-Uniform C&P']
        width = 0.75
        linear_local_ind = np.array([0, 4, 8])
        linear_fed_ind = linear_local_ind + width
        nonlinear_local_ind = linear_local_ind + 2*width
        nonlinear_fed_ind = linear_local_ind + 3*width
        tick_ind = np.mean([linear_local_ind, linear_fed_ind, nonlinear_local_ind, nonlinear_fed_ind], axis=0)

        # create axis labels
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_ylabel('Average Device Data Contribution', fontsize=20, weight='bold')
        ax1.bar(nonlinear_fed_ind, self.avg_data_fed[3:], width, color='tab:blue', label='Non-linear RealFM')
        ax1.bar(linear_fed_ind, self.avg_data_fed[:3], width, color='tab:red', label='Linear RealFM')
        ax1.bar(nonlinear_local_ind, self.avg_data_local[3:], width, color='tab:green', label='Local Non-linear')
        ax1.bar(linear_local_ind, self.avg_data_local[:3], width, color='tab:orange', label='Local Linear')

        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('log')

        plt.xticks(tick_ind, x, weight='bold', fontsize=20)
        ax1.grid(axis='y', alpha=0.25)
        ax1.set_yscale('symlog')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.tick_params(axis='y', which='major', labelsize=18)
        # plt.legend(fontsize=13, loc='upper left')

        if self.dataset == 'mnist':
            plt.ylim([0, 1e9])
        else:
            plt.ylim([0, 1e5])

        plt.tight_layout()

        if save_figure:
            title = 'realfm-average-device-contribution' + self.file_end
            if self.non_iid:
                title = title + str(self.dirichlet_value) + '.png'
            else:
                title = title + '.png'
            plt.savefig(title, dpi=200)
        else:
            plt.show()

    def test_accuracy_plot(self, save_figure, exp='uniform'):

        # get test accuracies
        fed_accs, local_accs = self.get_test_accuracy(exp)

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

        plt.legend(loc='lower right')
        plt.ylabel('Test Accuracy', fontsize=20, weight='bold')
        plt.xlabel('Epochs', fontsize=20, weight='bold')
        plt.xlim([1, self.epochs])
        plt.xscale("log")
        plt.grid(which="both", alpha=0.25)

        if self.dataset == 'mnist':
            plt.ylim([0.1, 1])
        else:
            plt.ylim([0.075, 0.85])

        # plt.tight_layout()
        if save_figure:
            savefilename = 'accuracy-comp' + self.file_end
            if self.non_iid:
                savefilename = savefilename + str(self.dirichlet_value) + '.png'
            else:
                savefilename = savefilename + '.png'
            plt.savefig(savefilename, dpi=200)
        else:
            plt.show()

    def get_test_accuracy(self, exp):
        fed_accs = []
        local_accs = []

        for trial in range(1, self.trials + 1):
            file = self.file_start + exp + '-run' + str(trial) + self.file_end
            fed_test_acc = unpack_data(file, self.epochs, self.num_workers, datatype='fed-epoch-acc-top1.log')
            local_test_acc = unpack_data(file, self.epochs, self.num_workers, datatype='local-epoch-acc-top1.log')
            fed_accs.append(fed_test_acc[:, 0])
            local_accs.append(np.mean(local_test_acc, axis=1))

        fed_accs = np.stack(fed_accs, axis=0)
        local_accs = np.stack(local_accs, axis=0)
        return fed_accs, local_accs

if __name__ == '__main__':
    clr = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']
    num_w = 16
    ds = 'mnist'
    non_iid = True
    dirichlet_value = 0.3

    plotter = RealFMPlotter(ds, num_w, clr, non_iid, dirichlet_value)
    # plotter.device_contribution_comparison(True)
    # plotter.server_utility_comparison(True)
    plotter.device_utility_comparison(True)
    # plotter.test_accuracy_plot(True)
