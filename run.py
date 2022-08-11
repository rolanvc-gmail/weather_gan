from abc import ABC

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from DMGR import DGMR
from data_modules import RadarDataset
from pytorch_lightning import Trainer, LightningDataModule
from torch.utils.data import DataLoader


class DGMRDataModule(LightningDataModule):
    def __init__(self):
        self.dataset = RadarDataset()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(self.dataset, batch_size=16)
        return dataloader


def main():
    datamodule = DGMRDataModule()
    model = DGMR()
    trainer = Trainer()
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()


