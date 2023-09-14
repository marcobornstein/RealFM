import torch
from utils.federated_communication import Communicator
from utils.data_loading import load_cifar10
from utils.equilibrium import optimal_data_local
from train_test import local_training, federated_training
from mpi4py import MPI
import numpy as np
import torchvision.models as models
from config import configs
from utils.recorder import Recorder

# split data up amongst 16 devices, then show how well a centralized model performs using 1-16
# averaged batches per update


if __name__ == '__main__':

    # determine config
    dataset = 'cifar10'
    config = configs[dataset]

    # determine hyper-parameters
    seed = config['random_seed']
    train_batch_size = config['train_bs']
    test_batch_size = config['test_bs']
    learning_rate = config['lr']
    epochs = config['epochs']
    log_frequency = config['log_frequency']

    # set seed for reproducibility
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    recorder = Recorder(config, dataset)

    # determine local data contributions
    # create different payoff functions as well (might multiply by a constant out front of the payoff function)
    # marginal_cost = np.random.uniform(9e-3, 0.0099)
    marginal_cost = 5e-3
    # marginal_cost = 1e-3
    optimal_updates_local = optimal_data_local(marginal_cost)

    print('rank: %d, optimal data: %d' % (rank, optimal_updates_local))

    # in order to partition data without overlap, share the amount of data each device will use
    device_num_data = np.empty(size, dtype=np.int32)
    comm.Allgather(np.array([optimal_updates_local], dtype=np.int32), device_num_data)

    # load CIFAR10 data
    trainloader, testloader = load_cifar10(device_num_data, rank, train_batch_size, test_batch_size)

    # use ResNet18
    model = models.resnet18()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # synchronize models (so they are identical initially)
    FLC.sync_models(model)

    # load model onto GPU (if available)
    model.to(device)

    # run local training (no federated mechanism)
    print('Beginning Training...')
    # local_training(model, trainloader, testloader, device, criterion, optimizer, epochs, log_frequency, recorder)
    federated_training(model, FLC, trainloader, testloader, device, criterion, optimizer, epochs, log_frequency,
                       recorder)

    print('Finished Training')

