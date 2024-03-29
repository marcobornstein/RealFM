import torch
import numpy as np
import torchvision.models as models
import copy
import argparse
from config import configs
from mpi4py import MPI
from utils.federated_communication import Communicator
from utils.data_loading import load_cifar10, load_mnist
from utils.equilibrium import optimal_data_local, optimal_data_fed, data_mapping
from train_test import local_training, federated_training, federated_training_nonuniform
from utils.recorder import Recorder
from utils.custom_models import MNIST


if __name__ == '__main__':

    # parse dataset from command line
    parser = argparse.ArgumentParser(description='RealFM Dataset Parser')
    parser.add_argument('--dataset', type=str, default='mnist')
    args = parser.parse_args()

    # determine config
    dataset = args.dataset
    config = configs[dataset]

    # determine hyper-parameters
    train_batch_size = config['train_bs']
    test_batch_size = config['test_bs']
    learning_rate = config['lr']
    epochs = config['epochs']
    log_frequency = config['log_frequency']
    marginal_cost = config['marginal_cost']
    local_steps = config['local_steps']
    uniform_payoff = config['uniform_payoff']
    uniform_cost = config['uniform_cost']
    linear_utility = config['linear_utility']
    a_opt = config['a_opt']
    k = config['k']
    simple_acc = config['simple_acc']
    non_iid = config['non_iid']
    alpha = config['dirichlet_value']
    num_data = config['num_train_data']
    og_marginal_cost = copy.deepcopy(marginal_cost)
    seed = config['random_seed']
    name = config['name']

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set seed for reproducibility
    torch.manual_seed(seed+rank)
    np.random.seed(seed+rank)

    # determine torch device available (default to GPU if available)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        dev = ["cuda:" + str(i) for i in range(num_gpus)]
        device = dev[gpu_id]

    else:
        num_gpus = 0
        device = "cpu"

    # initialize federated communicator
    FLC = Communicator(rank, size, comm, device)

    # initialize recorder
    recorder = Recorder(rank, size, config, name, dataset)

    if uniform_payoff:
        c = 1
    else:
        low = 0.9
        high = 1.1
        avg = (high+low)/2
        c = np.random.uniform(low, high)

    if uniform_cost:
        marginal_cost = marginal_cost
    else:
        marginal_cost = np.random.normal(marginal_cost, 0.05*marginal_cost)

    if uniform_payoff and uniform_cost:
        nu = False
    else:
        nu = True

    # keep note of the constant used
    recorder.save_payoff_c(marginal_cost)
    recorder.save_payoff_c(c)

    # determine local data contributions
    b_local, u_local = optimal_data_local(marginal_cost, c=c, k=k, a_opt=a_opt, linear=linear_utility,
                                          simple_acc=simple_acc)

    # map data down to boun
    max_data_per_device = num_data / size
    b_local_mapped = data_mapping(b_local, max_data_per_device)

    print('rank: %d, local optimal data: %d, marginal cost %f, payoff constant %f' % (rank, b_local_mapped, marginal_cost, c))

    # in order to partition data without overlap, share the amount of data each device will use
    device_num_data = np.empty(size, dtype=np.int32)
    comm.Allgather(np.array([b_local_mapped], dtype=np.int32), device_num_data)

    # determine self weight
    self_weight = b_local_mapped / np.sum(device_num_data)
    FLC.self_weight = self_weight

    # load CIFAR10 data
    if dataset == 'cifar10':
        trainloader, testloader = load_cifar10(device_num_data, rank, size, train_batch_size, test_batch_size, non_iid,
                                               alpha)
        model = models.resnet18()
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    elif dataset == 'mnist':
        trainloader, testloader = load_mnist(device_num_data, rank, size, train_batch_size, test_batch_size, non_iid,
                                             alpha)
        model = MNIST()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
    else:
        print('ERROR: Dataset Provided Is Not Valid.')
        exit()

    # use ResNet18
    criterion = torch.nn.CrossEntropyLoss()

    # synchronize models (so they are identical initially)
    FLC.sync_models(model)

    # save initial model for federated training
    model_path = recorder.saveFolderName + '-model.pth'
    if rank == 0:
        # torch.save(model.state_dict(), 'initial_weights.pth')
        torch.save(model, model_path)

    # load model onto GPU (if available)
    model.to(device)

    # run local training (no federated mechanism)
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('Beginning Local Training...')

    a_local = local_training(model, trainloader, testloader, device, criterion, optimizer,
                             epochs, log_frequency, recorder, scheduler)

    MPI.COMM_WORLD.Barrier()

    # reset model to the initial model
    model = torch.load(model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('Beginning Federated Training...')

    if not nu:
        a_fed = federated_training(model, FLC, trainloader, testloader, device, criterion, optimizer, epochs,
                                   log_frequency, recorder, scheduler, local_steps=local_steps)
    else:
        if uniform_payoff:
            b_local_uniform, _ = optimal_data_local(og_marginal_cost, c=1, k=k, a_opt=a_opt, linear=linear_utility,
                                                    simple_acc=simple_acc)
        else:
            b_local_uniform, _ = optimal_data_local(og_marginal_cost, c=avg, k=k, a_opt=a_opt, linear=linear_utility,
                                                    simple_acc=simple_acc)

        # need uniform number of steps to ensure no deadlock, use the expected (uniform) local data points
        b_local_uniform = data_mapping(b_local_uniform, max_data_per_device)
        steps_per_epoch = (b_local_uniform // train_batch_size) + 1
        a_fed = federated_training_nonuniform(model, FLC, trainloader, testloader, device, criterion, optimizer,
                                              steps_per_epoch, epochs, log_frequency, recorder, scheduler,
                                              local_steps=local_steps)

    MPI.COMM_WORLD.Barrier()

    # compute the optimal contributions that would've maximized utility
    b_fed = optimal_data_fed(a_local, a_fed, b_local_mapped, marginal_cost, c=c, linear=linear_utility)

    # print and store optimal amount of data
    print(f' [rank {rank}] initial local optimal data: {b_local_mapped}, federated mechanism optimal data: {b_fed}')
    recorder.save_data_contributions(b_local, b_local_mapped, b_fed)
