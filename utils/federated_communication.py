import torch
import time
from mpi4py import MPI


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


class Communicator:

    def __init__(self, rank, size, comm, device):
        self.comm = comm
        self.size = size
        self.rank = rank
        self.device = device
        self.tensor_list = list()
        self.send_buffer = None
        self.recv_buffer = None
        self.self_weight = None

    def average(self, state_dict):

        self.comm.Barrier()
        tic = time.time()

        state_dicts = MPI.COMM_WORLD.allgather(state_dict)

        self.comm.Barrier()
        toc = time.time()

        for i in range(len(state_dicts)):
            if i == self.rank:
                continue
            else:
                neighbor_sd = state_dicts[i]
                for key in state_dict.keys():
                    state_dict[key] += neighbor_sd[key]

        # when using self weight, leave commented
        # for key in state_dict.keys():
        #     state_dict[key] = torch.div(state_dict[key], self.size)

        return state_dict, toc - tic

    def sync_models(self, model):

        # prepare model to be communicated
        state_dict = self.prepare(model)

        # average models together
        for key in state_dict:
            state_dict[key] = state_dict[key] * self.self_weight
        state_dict, _ = self.average(state_dict)

        # reset local models to be the averaged model
        model.load_state_dict(state_dict)

    def prepare(self, model):
        if self.device == 'cpu':
            return model.state_dict()
        else:
            return {k: v.cpu() for k, v in model.state_dict().items()}

    def communicate(self, model):

        # prepare model to be communicated
        state_dict = self.prepare(model)

        # averaging across all device
        for key in state_dict:
            state_dict[key] = state_dict[key] * self.self_weight
        state_dict, comm_time = self.average(state_dict)

        # reset local models to be the averaged model
        model.load_state_dict(state_dict)

        return comm_time
