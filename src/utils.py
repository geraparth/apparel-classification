from sklearn.model_selection import train_test_split
import torch


def split_data(data_x, data_y, test_size=0.2):

    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size)
    return train_x, val_x, train_y, val_y


def convert_x_to_torch(np_data):

    rows = np_data.shape[0]
    columns = np_data.shape[1]
    np_data = np_data.reshape(rows, 1, columns, columns)
    tor_data_x = torch.from_numpy(np_data)

    return tor_data_x


def convert_y_to_torch(np_data):

    np_data = np_data.astype(int)
    tor_data_y = torch.from_numpy(np_data)

    return tor_data_y
