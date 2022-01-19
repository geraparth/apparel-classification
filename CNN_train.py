
from torch.autograd import Variable
import config.config as opt
from src.dataset import LoadDataset
from src import utils
from src.network import CNNNetwork

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch


class ModelTrain:

    def __init__(self):

        self.model = None
        self.train_losses = []
        self.val_losses = []

    def fit(self):

        x = LoadDataset()
        data_x, data_y = x.get_dataset('train')
        np_train_x, np_val_x, np_train_y, np_val_y = utils.split_data(data_x, data_y, 0.2)
        train_x = utils.convert_x_to_torch(np_train_x)
        train_y = utils.convert_y_to_torch(np_train_y)
        val_x = utils.convert_x_to_torch(np_val_x)
        val_y = utils.convert_y_to_torch(np_val_y)

        self.model = CNNNetwork()
        optimizer = Adam(self.model.parameters(), opt.learning_rate)
        criterion = CrossEntropyLoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            criterion = criterion.cuda()

        for i in range(opt.n_epochs):

            self.model.train()
            x_train, y_train = Variable(train_x), Variable(train_y)
            x_val, y_val = Variable(val_x), Variable(val_y)

            if torch.cuda.is_available():

                x_train = x_train.cuda()
                y_train = y_train.cuda()
                x_val = x_val.cuda()
                y_val = y_val.cuda()

            optimizer.zero_grad()

            output_train = self.model.forward_pass(x_train)
            output_val = self.model.forward_pass(x_val)

            loss_train = criterion(output_train, y_train)
            loss_val = criterion(output_val, y_val)

            self.train_losses.append(loss_train.detach().numpy())
            self.val_losses.append(loss_val.detach().numpy())

            loss_train.backward()
            optimizer.step()

            if opt.n_epochs % 2 == 0:
                print('Epoch: ', opt.n_epochs + 1, '\t', 'Training Loss: ', loss_train.detach().numpy(), '\t', 'Validation Loss: ', loss_val.detach().numpy())




