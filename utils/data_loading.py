import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from mpi4py import MPI


def non_iid_dirichlet(device_num_data, targets, alpha, size):
    # this dirichlet split follows from "FedDC: Federated Learning with Non-IID Data
    # via Local Drift Decoupling and Correction", although it has been altered for our purposes.

    # CIFAR10 has targets as list, so correct targets if its not a torch tensor
    targets = torch.tensor(targets) if isinstance(targets, list) else targets

    # determine number of classes
    num_classes = torch.max(targets) + 1

    # compute the index of each class within the overall dataset
    idx_list = [torch.where(targets == i)[0] for i in range(num_classes)]

    # compute dirichlet priors for each device (how much of each class is in a dataset for each device)
    cls_priors = np.random.dirichlet(alpha=[alpha]*num_classes, size=size)
    prior_cumsum = np.cumsum(cls_priors, axis=1)

    # counter for how much data in each class remains
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    # initialize the empty index list for each device
    device_indices = [np.empty(device_num_data[device]).astype(np.int32) for device in range(size)]

    while np.sum(device_num_data) != 0:
        # sample random device
        curr_device = np.random.randint(size)

        # choose new device if its data is already filled
        if device_num_data[curr_device] <= 0:
            continue

        # index will be added, so remove amount device needs by 1
        device_num_data[curr_device] -= 1

        # select the device's dirichlet prior
        curr_prior = prior_cumsum[curr_device]

        while True:

            # determine which data will be chosen given device's distribution of classes
            class_label = np.argmax(np.random.uniform() <= curr_prior)

            # choose new class if none remains
            if class_amount[class_label] <= 0:
                continue

            # remove class count by 1 since its now drawn and add index to device list of indices
            class_amount[class_label] -= 1
            device_indices[curr_device][device_num_data[curr_device]] = idx_list[class_label][class_amount[class_label]]
            break

    return device_indices


def data_partition(trainset, testset, train_batch_size, test_batch_size, num_data, rank, size, non_iid, alpha):
    global_data = trainset.data
    n_data = global_data.shape[0]

    # divvy up data according to amount of data each device needs
    num_data_cum_sum = np.cumsum(num_data)

    if num_data_cum_sum[-1] > len(trainset.targets):
        if rank == 0:
            print('ERROR: Not Enough Data for Given Marginal Costs')
            exit()

    # non-iid dirichlet split
    if non_iid:

        # have the root divvy up the data in the correct dirichlet manner for all devices
        if rank == 0:
            device_data_idxs = non_iid_dirichlet(num_data, trainset.targets, alpha, size)
        else:
            device_data_idxs = None

        # send all devices their data indices (broadcast from root)
        device_data_idxs = MPI.COMM_WORLD.bcast(device_data_idxs, root=0)

        # each device takes their own set of data indices
        device_data_idx = device_data_idxs[rank]

    else:
        # spread data out iid amongst devices
        data_idx = np.arange(n_data)
        np.random.shuffle(data_idx)

        if rank == 0:
            s_idx = 0
        else:
            s_idx = num_data_cum_sum[rank - 1]
        e_idx = num_data_cum_sum[rank]
        device_data_idx = data_idx[s_idx:e_idx]

    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[device_data_idx]
    trainset.data = trainset.data[device_data_idx, :, :]

    # load train and test data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    return trainloader, testloader


def load_mnist(num_data, rank, size, train_batch_size, test_batch_size, non_iid, alpha):

    # transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load mnist data
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST('../data', train=False, transform=transform)
    trainloader, testloader = data_partition(trainset, testset, train_batch_size, test_batch_size, num_data, rank,
                                             size, non_iid, alpha)
    return trainloader, testloader


def load_cifar10(num_data, rank, size, train_batch_size, test_batch_size, non_iid, alpha):

    # get CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    trainloader, testloader = data_partition(trainset, testset, train_batch_size, test_batch_size, num_data, rank,
                                             size, non_iid, alpha)
    return trainloader, testloader
