import torch
from torch import optim
from model import DNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from utils.evaluation import *
from utils.seq_parser import *
from utils.dataset import *
import copy
import os
from functools import partial



def double_print(text, output_file = None, end = '\n'):
    print(text, end = end)
    if not output_file is None:
        output_file.write(str(text) + end)
        output_file.flush()

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class Solver(object):
    def __init__(self, config):
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.model_path = config.model_path
        self.logs_path = config.logs_path
        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.lr_update_mode = config.lr_update_mode
        self.lr_schedule = config.lr_schedule

        self.save_every = config.save_every
        self.seed = config.seed

        self.hidden_size = config.hidden_size
        self.valid_ratio = config.valid_ratio

        self.differ_data = config.differ_data

        self.lr_func = continuous_seq(**self.lr_schedule) if self.lr_schedule != None else None
        self.criterion = nn.CrossEntropyLoss()
        self.model = DNN(self.hidden_size)
        self.optimizer = optim.SGD(self.model.parameters(), self.lr)
        self.build_model()
        self.best_acc = 0
        self.best_epoch = 0
        self.best_model = self.model

        self.epsilon_test = config.epsilon_test
        self.eps_iter_test = config.eps_iter_test
        self.nb_iter_test = config.nb_iter_test

        if self.dataset == 'MNIST':
            self.train_batches = (60000 - 1) // self.batch_size + 1
            self.train_loader, self.valid_loader, self.test_loader, classes = mnist(batch_size=self.batch_size,
                                                                     valid_ratio=self.valid_ratio)
        elif self.dataset == 'FashionMNIST':
            self.train_batches = (60000 - 1) // self.batch_size + 1
            self.train_loader, self.valid_loader, self.test_loader, classes = FashionMNIST(batch_size=self.batch_size,
                                                                     valid_ratio=self.valid_ratio)
        elif self.dataset == "cifar10":
            self.train_batches = (10000 - 1) // self.batch_size + 1
            self.train_loader, self.valid_loader, self.test_loader, classes = cifar10(batch_size=self.batch_size,
                                                                     valid_ratio=self.valid_ratio)

    def build_model(self):
        output_file = open(f"{self.logs_path}/log.txt", 'w')
        file_print = partial(double_print, output_file = output_file)

        torch.manual_seed(self.seed)
        self.model.apply(self.weights_init)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        file_print("Total params: {}".format(self.num_params))
        if torch.cuda.is_available():
            self.model.cuda()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, np.sqrt(1.9822**2/ m.in_features))
            # m.weight.data.normal_(0.0, 2.0 / (m.out_features * m.out_features))
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, np.sqrt(1.9822**2 / m.in_channels))

    def train(self):
        output_file = open(f"{self.logs_path}/log.txt", 'w')
        file_print = partial(double_print, output_file = output_file)

        acc_calculator = AverageCalculator()
        loss_calculator = AverageCalculator()

        total_dataset = None
        total_labels = None
        for i, (data, labels) in enumerate(self.train_loader):
            if i == 0:
                total_dataset = data.numpy()
                total_labels = labels.numpy()
            else:
                total_dataset = np.concatenate((total_dataset, data.numpy()), axis = 0)
                total_labels = np.concatenate((total_labels, labels.numpy()), axis = 0)
        np.save(f"{self.logs_path}/total_dataset.npy", total_dataset)
        np.save(f"{self.logs_path}/total_labels.npy", total_labels)

        for epoch in range(self.num_epochs):
            acc_calculator.reset()
            loss_calculator.reset()

            self.model.train()

            temp_data_size = 0
            flag = False
            for i, (data, labels) in enumerate(self.train_loader):
                temp_data_size += data.size(0)
                if self.differ_data!=None and flag == False and temp_data_size >= self.differ_data:
                    differ_data_value = data[self.differ_data - temp_data_size + data.size(0) - 1]
                    np.save(f"{self.logs_path}/differ_data.npy", differ_data_value.numpy())
                    data = torch.cat([data[0:self.differ_data - temp_data_size + data.size(0) - 1], data[self.differ_data - temp_data_size + data.size(0):]])
                    labels = torch.cat([labels[0:self.differ_data - temp_data_size + data.size(0) - 1], labels[self.differ_data - temp_data_size + data.size(0):]])
                    flag = True

                epoch_batch_idx = epoch + 1. / self.train_batches * i if self.lr_update_mode.lower() in ['batch',] else epoch

                # Update the learning rate
                lr_this_batch = self.lr_func(epoch_batch_idx)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_batch
                if i == 0:
                    file_print('Learning rate = %1.2e' % lr_this_batch)

                data = data.type(torch.FloatTensor)
                data = to_cuda(data)
                labels = to_cuda(labels)

                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.criterion(pred, labels)
                acc = accuracy(pred, labels)
                loss.backward()
                self.optimizer.step()

                loss_calculator.update(loss.item(), data.size(0))
                acc_calculator.update(acc.item(), data.size(0))

            loss_this_epoch = loss_calculator.average
            acc_this_epoch = acc_calculator.average
            file_print('Train loss / acc after epoch %d: %.4f / %.2f%%' % ((epoch + 1), loss_this_epoch, acc_this_epoch * 100.))

            loss_calculator.reset()
            acc_calculator.reset()

            self.model.eval()

            for i, (data_test, labels_test) in enumerate(self.test_loader):
                data_test = data_test.type(torch.FloatTensor)
                data_test = to_cuda(data_test)
                labels_test = to_cuda(labels_test)

                data_test.requires_grad = True
                self.optimizer.zero_grad()
                pred_test = self.model(data_test)
                loss_test = self.criterion(pred_test, labels_test)
                acc_test = accuracy(pred_test, labels_test)
                loss_test.backward()
                grad_mat = data_test.grad.view(-1, 32 * 32 * 3)
                grad_vec = (grad_mat * grad_mat).sum()
                loss_calculator.update(loss_test.item(), data_test.size(0))
                acc_calculator.update(acc_test.item(), data_test.size(0))




            loss_this_epoch = loss_calculator.average
            acc_this_epoch = acc_calculator.average
            file_print('Test loss / acc after epoch %d: %.4f / %.2f%%' % ((epoch + 1), loss_this_epoch, acc_this_epoch * 100.))

            if acc_this_epoch >= self.best_acc:
                self.best_acc = acc_this_epoch
                self.best_epoch = epoch
                self.best_model = self.model


            if (epoch + 1) % self.save_every == 0:
                CCP_model_path = os.path.join(self.model_path, 'model-{}.pkl'.format(epoch+1))
                torch.save(self.model, CCP_model_path)

        CCP_model_path = os.path.join(self.model_path, 'model-best.pkl')
        torch.save(self.best_model, CCP_model_path)
        file_print('Best acc is after epoch %d: %.2f%%' % ((self.best_epoch + 1), self.best_acc * 100.))