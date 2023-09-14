import numpy as np
import os


class Recorder(object):
    def __init__(self, config, dataset):
        self.record_comp_times = list()
        self.record_comm_times = list()
        self.record_losses = list()
        self.record_training_acc = list()
        self.record_test_acc = list()
        self.epoch_test_acc = list()
        self.saveFolderName = config['file_path'] + '/' + config['name'] + '-' + dataset

        if not os.path.isdir(self.saveFolderName):
            os.mkdir(self.saveFolderName)
            with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                f.write(str(config) + '\n')
        else:
            self.saveFolderName = self.saveFolderName + '1'
            os.mkdir(self.saveFolderName)
            with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                f.write(str(config) + '\n')

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
            np.savetxt(self.saveFolderName + '/epoch-acc-top1.log', self.epoch_test_acc, delimiter=',')
        else:
            self.record_test_acc.append(test_acc)
            np.savetxt(self.saveFolderName + '/test-acc-top1.log', self.record_test_acc, delimiter=',')

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/comp-time.log', self.record_comp_times, delimiter=',')
        np.savetxt(self.saveFolderName + '/comm-time.log', self.record_comm_times, delimiter=',')
        np.savetxt(self.saveFolderName + '/train-loss.log', self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName + '/train-acc-top1.log', self.record_training_acc, delimiter=',')