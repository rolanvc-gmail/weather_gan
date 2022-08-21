from abc import ABC

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from DMGR import DGMR
from data_modules import RadarDataset
from pytorch_lightning import Trainer, LightningDataModule
from torch.utils.data import DataLoader



def main():
    dgmr = DGMR()
    radar_dataset = RadarDataset()
    train_dataloader = DataLoader(radar_dataset, batch_size=16, shuffle=True)

    for b in range(2000):
        print("Step # {}".format(b))
        batch_data = next(iter(train_dataloader))
        dgmr.training_step(batch_data)



if __name__ == "__main__":
    main()


