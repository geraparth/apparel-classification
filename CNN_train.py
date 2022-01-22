
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
from datetime import date

import config.config as opt
from src.dataset import LoadDataset
from src import utils
from src.network import CNNNetwork

#*******************************************************


class ModelTrain:

    def __init__(self):

        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.train_accuracy = None
        self.val_accuracy = None
        self.sample_submission = pd.read_csv(os.path.join(opt.data_root, opt.submission))

#*********************************************************************

    def fit(self):
        """
            Function to fit CNN model to the training data.

            Parameters: self - instance of the class


            Returns:
                    Saves
                    self.model : Training model
                    self.train_losses : list variable containing train loss for each epoch
                    self.val_losses : list variable containing val loss for each epoch

                    """

        x = LoadDataset()
        data_x, data_y = x.get_dataset('train')
        np_train_x, np_val_x, np_train_y, np_val_y = utils.split_data(data_x, data_y, 0.2)
        self.train_x = utils.convert_x_to_torch(np_train_x)
        self.train_y = utils.convert_y_to_torch(np_train_y)
        self.val_x = utils.convert_x_to_torch(np_val_x)
        self.val_y = utils.convert_y_to_torch(np_val_y)

        self.model = CNNNetwork()
        optimizer = Adam(self.model.parameters(), opt.learning_rate)
        criterion = CrossEntropyLoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            criterion = criterion.cuda()

        for i in range(opt.n_epochs):

            self.model.train()
            x_train, y_train = Variable(self.train_x), Variable(self.train_y)
            x_val, y_val = Variable(self.val_x), Variable(self.val_y)

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

            if i % 2 == 0:
                print('Epoch: ', opt.n_epochs + 1, '\t', 'Training Loss: ', loss_train.detach().numpy(), '\t', 'Validation Loss: ', loss_val.detach().numpy())

#****************************************************************
    def plot_loss_curve(self):

        """
            Function to fit CNN model to the training data.

            Parameters: self - instance of the class

            Returns:
                    saves train error/val loss Vs epoch graph

                            """

        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.legend()
        plt.savefig(opt.graph_root + str(date.today()) + '.png')
        plt.show()


#****************************************************************

    def get_train_accuracy(self):

        """
            Function to calculate train accuracy of the model.

            Parameters: self - instance of the class

            Returns:
                    saves train accuracy in the variable self.train_accuracy

                                    """

        with torch.no_grad():
            output = self.model.forward_pass(self.train_x)

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        self.train_accuracy = accuracy_score(self.train_y, predictions)

#****************************************************************

    def get_val_accuracy(self):

        """
        Function to calculate val accuracy of the model.

        Parameters: self - instance of the class

        Returns:
                saves validation accuracy in the variable self.val_accuracy

                                            """

        with torch.no_grad():
            output = self.model.forward_pass(self.val_x)

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        self.val_accuracy = accuracy_score(self.val_y, predictions)

#****************************************************************

    def get_test_result(self):

        """
        Function to make predictions on the test and save.

        Parameters: self - instance of the class

        Returns:
                saves test predictions in the file sample submission

                                                    """
        x = LoadDataset()
        data_x = x.get_dataset('test')
        test_x = utils.convert_x_to_torch(data_x)

        with torch.no_grad():
            output = self.model.forward_pass(test_x)

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        self.sample_submission['label'] = predictions
        self.sample_submission.head()
        self.sample_submission.to_csv(opt.prediction_root + str(date.today())+'.csv', index=False)

#****************************************************************

    def save_model(self):

        """
        Function to save model in directory.

        Parameters: self - instance of the class

        Returns:
                saves model

        """

        torch.save(self.model.state_dict(), os.path.join(opt.model_path,'model'+str(date.today())))


apparel_classification = ModelTrain()
apparel_classification.fit()

apparel_classification.get_train_accuracy()
apparel_classification.get_val_accuracy()


print(apparel_classification.train_accuracy)
print(apparel_classification.val_accuracy)

apparel_classification.get_test_result()

apparel_classification.plot_loss_curve()
apparel_classification.save_model()



