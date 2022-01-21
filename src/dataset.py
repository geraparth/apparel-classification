import config.config as opt
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import random
from skimage.io import imread


class LoadDataset:

    # assertion_sample = 5

    # def __init__(self):
    def read_image(self, image_urls, dataset_type='train'):

        # sample = min(len(image_urls), self.assertion_sample)
        # assert isinstance(image_urls, list), "Image url must be a list of valid Image URL's"
        # assert all([isinstance(p, str) for p in random.sample(image_urls, sample)]), "Image url must be a str data type"
        # assert all([p.split(".")[-1] in opt.acceptable_img_format for p in random.sample(image_urls, sample)]), f"Image url must end in {opt.acceptable_img_format}"

        if dataset_type == 'train':
            image_path = os.path.join(opt.data_root, opt.train_images)
        else:
            image_path = os.path.join(opt.data_root, opt.test_images)

        image_array = []

        for image in tqdm(image_urls):
            img = imread(os.path.join(image_path, str(image) + '.png'), as_gray=True)
            img /= 255.0
            img = img.astype('float32')
            image_array.append(img)

        data_x = np.array(image_array)

        return data_x

    def get_dataset(self, dataset_type='train'):

        if dataset_type == 'train':

            image_urls = list(pd.read_csv(os.path.join(opt.data_root, opt.train_images) + '.csv')['id'])
            data_y = np.array(pd.read_csv(os.path.join(opt.data_root, opt.train_images) + '.csv')['label'])
            data_x = self.read_image(image_urls, dataset_type)
            return data_x, data_y

        else:
            image_urls = list(pd.read_csv(os.path.join(opt.data_root, opt.test_images) + '.csv')['id'])
            data_x = self.read_image(image_urls, dataset_type)
            return data_x

