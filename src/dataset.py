import config.config as opt
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from skimage.io import imread


class LoadDataset:

    #def __init__(self):
    def read_image(self, image_urls, dataset_type='train'):

        if dataset_type == 'train':
            image_path = os.path.join(opt.data_root, opt.train_images)
        else:
            image_path = os.path.join(opt.data_root, opt.test_images)

        image_array = []

        for image in tqdm(image_urls):

            img = imread(image_path+str(image)+'.png', as_gray=True)
            img /= 255.0
            img = img.astype('float32')
            image_array.append(img)

        data_x = np.array(image_array)
        data_y = np.array(pd.read_csv(image_path+'.csv')['label'])

        return data_x, data_y

    def get_dataset(self, dataset_type='train'):

        if dataset_type == 'train':

            image_urls = list(pd.read_csv(os.path.join(opt.data_root, opt.train_images) + '.csv')['id'])

        else:
            image_urls = list(pd.read_csv(os.path.join(opt.data_root, opt.test_images) + '.csv')['id'])

        data_x, data_y = self.read_image(image_urls, dataset_type)
        return data_x, data_y

