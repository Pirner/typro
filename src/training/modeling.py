import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from timm import create_model
from torchmetrics import Accuracy


class PawsModel(pl.LightningModule):
    def __init__(self, model_name='convnext_tiny', dropout=0.1):
        super(PawsModel, self).__init__()
        self.model_name = model_name
        self.backbone = create_model(self.model_name, pretrained=True, num_classes=1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(1)
        self.sig = nn.Sigmoid()
        self.mse = nn.MSELoss()
        self.criterion = nn.BCELoss()

        self.test_preds = []
        self.accuracy = Accuracy(task="binary", num_classes=2)

    def RMSE(self, preds, y):
        mse = self.mse(preds.view(-1), y.view(-1))
        return torch.sqrt(mse)

    def forward(self, sample):
        x = sample
        x = self.backbone(x)
        x = self.drop(x)
        # cat = torch.cat([x, m], dim=1)
        logit = self.fc(x)
        logit = self.sig(logit)
        return logit

    def training_step(self, batch, batch_idx):
        sample, y = batch

        preds = self(sample)

        # loss = self.RMSE(preds, y)
        y_unsq = torch.unsqueeze(y, -1)
        loss = self.criterion(preds, y_unsq)
        acc = self.accuracy(preds, y_unsq)
        self.log('train_loss', loss.item(), on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sample, y = batch

        preds = self(sample)
        y_unsq = torch.unsqueeze(y, -1)

        loss = self.criterion(preds, y_unsq)
        acc = self.accuracy(preds, y_unsq)

        self.log('val', acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

    def test_step(self, batch, batch_idx):
        sample = batch
        preds = 100 * self(sample)
        self.test_preds.append(preds.detach().cpu())

    def get_predictions(self):
        return torch.cat(self.test_preds).numpy()
