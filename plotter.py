import os
import matplotlib.pyplot as plt
import numpy as np


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
    epochs = 100
    fed_accs = []
    local_accs = []
    iters = np.arange(1, epochs + 1)
    each_dev_local_a = []

    for trial in range(1, 4):
        file = 'output/uniform/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
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

    # =======================================================
    # plot data contribution plot and bar chart

    fed_a = y_mean[-1] * np.ones(nw)
    local_b = np.empty((3, nw))
    fed_b = np.empty((3, nw))
    for trial in range(1, 4):

        file = 'output/uniform/realfm-uniform-run' + str(trial) + '-cifar10-' + str(nw) + 'devices'
        optimal_data = unpack_data(file, 2, nw, datatype='update-contribution.log')
        local_b[trial-1, :] = optimal_data[0, :]
        fed_b[trial-1, :] = optimal_data[1, :]

    local_a = np.mean(np.stack(each_dev_local_a, axis=0), axis=0)
    local_b = np.mean(local_b, axis=0)
    fed_b = np.mean(fed_b, axis=0)

    # scatter plot
    '''
    plt.figure()
    # plt.scatter(local_b, local_a, color='r')
    # plt.scatter(fed_b, fed_a, color='b')
    for i in range(len(local_b)):
        x = np.array([local_b[i], fed_b[i]])
        y = np.array([local_a[i], fed_a[i]])
        plt.plot(x, y)

    plt.ylabel('Test Accuracy', fontsize=13)
    plt.xlabel('Update Contribution ($b_i$)', fontsize=13)
    plt.grid(which="both", alpha=0.25)
    # plt.xscale("symlog")
    plt.show()
    '''

    # bar plot
    # '''
    added_b = fed_b - local_b
    devices = np.arange(1, nw+1)
    plt.figure()
    plt.bar(devices, local_b, label='Local Optimal Contributions $(b_i^o)$')
    plt.bar(devices, added_b, bottom=local_b, label='Incentivized Contributions $(b_i^* - b_i^o)$')
    plt.xlabel('Devices', fontsize=13)
    plt.ylabel('Update Contribution ($b_i$)', fontsize=13)
    plt.xticks(devices)
    plt.legend(loc='lower left', fontsize=10)
    # plt.ylim([0, 5500])
    savefilename2 = str(nw) + 'device-bar' + '.png'
    plt.savefig(savefilename2, dpi=200)
    # plt.show()
    # '''
