from mpi4py import MPI
import numpy as np
import os
import datetime


def date_string(date):
    return date.day * 1000000 + date.hour*10000 + date.minute*100 + date.second



class Recorder(object):
    def __init__(self, rank, size, config, name, dataset):
        self.rank = rank
        self.size = size

        # local resultss
        self.record_comp_times = list()
        self.record_comm_times = list()
        self.record_losses = list()
        self.record_training_acc = list()
        self.epoch_test_acc = list()

        # fed results
        self.record_comp_times_f = list()
        self.record_comm_times_f = list()
        self.record_losses_f = list()
        self.record_training_acc_f = list()
        self.epoch_test_acc_f = list()

        # remaining
        self.record_test_acc = list()
        self.update_contribution = list()
        self.saveFolderName = config['file_path'] + '/' + name + '-' + dataset + '-' + str(size) + 'devices'

        if rank == 0:
            if not os.path.isdir(self.saveFolderName):
                flag = np.array([0, 0], dtype=np.int32)
                os.mkdir(self.saveFolderName)
                with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                    f.write(str(config) + '\n')
            else:
                current_date = datetime.datetime.now()
                val = date_string(current_date)
                add_on = "-" + str(date_string(current_date))
                flag = np.array([1, int(val)], dtype=np.int32)
                self.saveFolderName = self.saveFolderName + add_on
                os.mkdir(self.saveFolderName)
                with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                    f.write(str(config) + '\n')
            MPI.COMM_WORLD.Bcast(flag, root=0)

        else:
            flag = np.empty(2, dtype=np.int32)
            MPI.COMM_WORLD.Bcast(flag, root=0)

        if flag[0]:
            add_on = str(flag[1])
            if rank > 0:
                self.saveFolderName = self.saveFolderName + '-' + add_on

        MPI.COMM_WORLD.Barrier()

    def get_save_folder(self):
        return self.saveFolderName

    def add_new(self, comp_time, comm_time, train_acc1, losses, local=True):
        if local:
            self.record_comp_times.append(comp_time)
            self.record_comm_times.append(comm_time)
            self.record_training_acc.append(train_acc1)
            self.record_losses.append(losses)
        else:
            self.record_comp_times_f.append(comp_time)
            self.record_comm_times_f.append(comm_time)
            self.record_training_acc_f.append(train_acc1)
            self.record_losses_f.append(losses)

    def add_test_accuracy(self, test_acc, epoch=True, local=True):
        if epoch:
            if local:
                self.epoch_test_acc.append(test_acc)
                np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-local-epoch-acc-top1.log',
                           self.epoch_test_acc, delimiter=',')
            else:
                self.epoch_test_acc_f.append(test_acc)
                np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-fed-epoch-acc-top1.log',
                           self.epoch_test_acc_f, delimiter=',')
        else:
            self.record_test_acc.append(test_acc)
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-acc-top1.log', self.record_test_acc,
                       delimiter=',')

    def save_to_file(self, local=True):
        if local:
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-local-comp-time.log',
                       self.record_comp_times, delimiter=',')
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-local-comm-time.log',
                       self.record_comm_times, delimiter=',')
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-local-train-loss.log',
                       self.record_losses, delimiter=',')
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-local-train-acc-top1.log',
                       self.record_training_acc, delimiter=',')
        else:
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-fed-comp-time.log', self.record_comp_times_f,
                       delimiter=',')
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-fed-comm-time.log', self.record_comm_times_f,
                       delimiter=',')
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-fed-train-loss.log', self.record_losses_f,
                       delimiter=',')
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-fed-train-acc-top1.log',
                       self.record_training_acc_f, delimiter=',')

    def save_data_contributions(self, b_local, b_local_mapped, b_fed):
        self.update_contribution.append(b_local)
        self.update_contribution.append(b_local_mapped)
        self.update_contribution.append(b_fed)
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-update-contribution.log', self.update_contribution,
                   delimiter=',')

    def save_payoff_c(self, c):
        self.update_contribution.append(c)
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-update-contribution.log', self.update_contribution,
                   delimiter=',')
