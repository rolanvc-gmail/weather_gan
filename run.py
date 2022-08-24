from abc import ABC

from DMGR import DGMR
from data_modules import RadarDataset
from torch.utils.data import DataLoader



def main():
    dgmr = DGMR().cuda()
    radar_dataset = RadarDataset()

    train_dataloader = DataLoader(radar_dataset, batch_size=16, shuffle=True)

    for b in range(2000):
        print("Step # {}".format(b))
        batch_data = next(iter(train_dataloader))
        dgmr.training_step(batch_data)



if __name__ == "__main__":
    main()


