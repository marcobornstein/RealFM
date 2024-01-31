import time
from mpi4py import MPI


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
