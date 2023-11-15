import os
import glob
from typing import List

from src.data.DTO import DataPoint


class DataIO:
    @staticmethod
    def read_data_points(dataset_path: str) -> List[DataPoint]:
        """
        searches for all data points for Tyre quality classification within the dataset
        :param dataset_path: path to the dataset to create data points from
        :return:
        """
        ret = []
        im_paths = glob.glob(os.path.join(dataset_path, '**/*.jpg'), recursive=True)
        for im_p in im_paths:
            if 'good' in im_p:
                label = 0
            else:
                label = 1
            data_point = DataPoint(im_path=im_p, label=label)
            ret.append(data_point)

        return ret
