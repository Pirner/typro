from typing import List

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
from PIL import Image
import cv2

from src.data.DTO import DataPoint


class TyreClassificationDataset(Dataset):
    def __init__(
            self,
            data=List[DataPoint],
            mean=0,
            std=1,
            transform=None
    ):
        """
        dataset for water segmentation in flooded areas
        :param data: data points to create train classifier from
        :param mean: mean value for standardization
        :param std: standard deviation
        :param transform: transformation to run
        """
        self.data_points = data
        self.mean = mean
        self.std = std

        self.transform = transform

    def __len__(self):
        """
        length of the dataset
        :return:
        """
        return len(self.data_points)

    def __getitem__(self, idx):
        """
        get an item from the data loading
        :param idx: index to load item from
        :return:
        """
        data_point = self.data_points[idx]

        img = cv2.imread(data_point.im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = data_point.label
        label = np.array(label)

        # mask_building_flooded = mask == 1
        # mask_road_flooded = mask == 3
        # mask_water = mask == 5
        # mask = mask_road_flooded.astype(int) + mask_building_flooded.astype(int) + mask_water.astype(int)
        #
        # mask = mask.astype(float)
        # mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            aug = self.transform(image=img)
            img = Image.fromarray(aug['image'])

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), ])
        img = t(img)
        label = torch.from_numpy(label).long()

        label = label.type(torch.FloatTensor)

        return img, label
