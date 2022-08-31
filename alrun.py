from abc import ABC

from alDMGR import AlDGMR
from data_modules import RadarDataset
from torch.utils.data import DataLoader



def main():
    print("********************")
    print("*** Running AL Code")
    print("********************")
    dgmr = AlDGMR().cuda()
    radar_dataset = RadarDataset()

    train_dataloader = DataLoader(radar_dataset, batch_size=16, shuffle=True)

    for b in range(2):
        print("Step # {}".format(b))
        batch_data = next(iter(train_dataloader))
        dgmr.training_step(batch_data)



if __name__ == "__main__":
    main()


