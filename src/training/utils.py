import random
from typing import List, Tuple

from src.data.DTO import DataPoint


class TyproTrainingUtils:
    @staticmethod
    def split_train_val(data: List[DataPoint], val_size=0.2) -> Tuple[List[DataPoint], List[DataPoint]]:
        """
        split the data into train and validation data
        :param data: data to split
        :param val_size: validation size from the data
        :return:
        """
        goods = list(filter(lambda x: x.label == 0, data))
        defects = list(filter(lambda x: x.label == 1, data))

        n_val_defects = int(len(defects) * val_size)
        n_val_goods = int(len(goods) * val_size)

        val_goods = goods[:n_val_goods]
        train_goods = goods[n_val_goods:]

        val_defects = defects[:n_val_defects]
        train_defects = defects[n_val_defects:]

        train_data = train_goods + train_defects
        val_data = val_goods + val_defects

        random.shuffle(train_data)
        random.shuffle(val_data)

        return train_data, val_data
