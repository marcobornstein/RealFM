import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms


def load_cifar10(num_data, rank, train_batch_size, test_batch_size):

    # get CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    global_data = trainset.data
    n_data = global_data.shape[0]

    # spread data out iid amongst devices
    data_idx = np.arange(n_data)
    np.random.shuffle(data_idx)

    # divvy up data according to amount of data each device needs
    num_data_cum_sum = np.cumsum(num_data)

    if num_data_cum_sum[-1] > 50000:
        if rank == 0:
            print('ERROR: Not Enough CIFAR10 Data for Given Marginal Costs')
            exit()

    if rank == 0:
        s_idx = 0
    else:
        s_idx = num_data_cum_sum[rank-1]
    e_idx = num_data_cum_sum[rank]
    device_data_idx = data_idx[s_idx:e_idx]

    # old method when all devices use equal partitions
    # device_data_idx = np.array_split(data_idx, size)[rank]

    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[device_data_idx]
    trainset.data = trainset.data[device_data_idx, :, :, :]

    # load train and test data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return trainloader, testloader
