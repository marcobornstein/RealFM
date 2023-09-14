from mpi4py import MPI
import numpy as np
import os
import datetime


def date_string(date):
    return date.day * 1000000 + date.hour*10000 + date.minute*100 + date.second



class Recorder(object):
    def __init__(self, rank, config, dataset):
        self.rank = rank
        self.record_comp_times = list()
        self.record_comm_times = list()
        self.record_losses = list()
        self.record_training_acc = list()
        self.record_test_acc = list()
        self.epoch_test_acc = list()
        self.saveFolderName = config['file_path'] + '/' + config['name'] + '-' + dataset

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

    def add_new(self, comp_time, comm_time, train_acc1, losses):
        self.record_comp_times.append(comp_time)
        self.record_comm_times.append(comm_time)
        self.record_training_acc.append(train_acc1)
        self.record_losses.append(losses)

    def add_test_accuracy(self, test_acc, epoch=False):
        if epoch:
            self.epoch_test_acc.append(test_acc)
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-acc-top1.log', self.epoch_test_acc, delimiter=',')
        else:
            self.record_test_acc.append(test_acc)
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-acc-top1.log', self.record_test_acc, delimiter=',')

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comp-time.log', self.record_comp_times, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comm-time.log', self.record_comm_times, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-loss.log', self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-acc-top1.log', self.record_training_acc, delimiter=',')
