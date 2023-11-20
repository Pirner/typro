import random

import torch
from transformers import ConvNextConfig, ConvNextModel
import torch.optim as optim
import torch.nn as nn
import timm
from timm import create_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data.io import DataIO
from src.training.generation import TyreClassificationDataset
from src.data.transformation import TransformerConfig
from src.training.modeling import PawsModel
from src.training.utils import TyproTrainingUtils


def main():
    dataset_path = r'C:\data\typro\archive\Digital images of defective and good condition tyres'
    data = DataIO.read_data_points(dataset_path)
    train_data, val_data = TyproTrainingUtils.split_train_val(data)
    print('[INFO] found {0} data points to train on'.format(len(data)))
    train_ds = TyreClassificationDataset(
        data=train_data,
        transform=TransformerConfig.get_train_transforms(im_h=224, im_w=224),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=1
    )
    val_ds = TyreClassificationDataset(
        data=val_data,
        transform=TransformerConfig.get_train_transforms(im_h=224, im_w=224),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=2,
        num_workers=1
    )
    # m = create_model('convnext_tiny', pretrained=True, num_classes=1)
    # print(m.head)
    model = PawsModel()
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=6,
                         callbacks=[
                             EarlyStopping(monitor="val_loss",
                                           mode="min",
                                           patience=2,
                                           )
                         ]
                         )
    trainer.fit(model, train_loader, val_loader)
    print('Finished Training')


if __name__ == '__main__':
    main()
