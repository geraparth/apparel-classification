from sklearn.model_selection import train_test_split
import torch

#**************************************************

def split_data(data_x, data_y, sample_size=0.2):
    """
            Function to split data into the training set and validation set.

            Parameters:
                    data_x : A numpy array containing the image RGBs
                    data_y : A 1-d array containing the image label
                    sample_size : The proportion of test set required

            Returns:
                    train_x : np array containing image RGBs - Training 
                    val_x : np array containing image RGBs - Validation
                    train_y :
                    val_y :

            """

    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=sample_size)
    return train_x, val_x, train_y, val_y

#****************************************************

def convert_x_to_torch(np_data):
    """
            Function that converts numpy image data to torch tensor

            Parameters:
                    np_data : A numpy array containing the image RGBs

            Returns:
                    tor_data_x : Image RGBs converted to torch

                """

    rows = np_data.shape[0]
    columns = np_data.shape[1]
    np_data = np_data.reshape(rows, 1, columns, columns)
    tor_data_x = torch.from_numpy(np_data)

    return tor_data_x

#*******************************************************

def convert_y_to_torch(np_data):

    """
                Function that converts image labels to torch tensor

                Parameters:
                        np_data : A 1D array containing the image labels

                Returns:
                        tor_data_y : Image labels converted to torch

                    """

    np_data = np_data.astype(int)
    tor_data_y = torch.from_numpy(np_data)

    return tor_data_y
