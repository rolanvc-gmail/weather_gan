import random
import os
import glob
import numpy as np
from torch.utils.data.dataset import Dataset
from data.data_stats import gather_stats
from process_data import dest_folder_npy
from typing import Any, Tuple
from einops import rearrange


class RadarDataset(Dataset):
    def __init__(self):
        self.base_folder = dest_folder_npy
        self.stats = gather_stats(self.base_folder)
        self.num_datapoints = self.stats["total_datapoints_all"]
        self.months = []
        for k in self.stats.keys():
            if k != "total_datapoints_all":
                self.months.append(k)
        self.months = sorted(self.months)

    def __len__(self):
        return self.num_datapoints

    def get_month_from_idx(self, idx):
        """
        get which month the idx most likely falls in.
        :param idx:
        :return:
        """
        running_total = 0

        for m in self.months:
            total_datapoints_in_months = self.stats[m]['total_datapoints']
            if idx < running_total + total_datapoints_in_months:
                return m, running_total
            else:
                running_total += total_datapoints_in_months
        raise Exception("item index is {} while total datapoints is {}.".format(idx, self.num_datapoints))

    def get_day_from_idx_and_month(self, idx: int, month: int, previous_months_total: int) -> Tuple[Any, int]:
        """ Given the item idx, the month and the previous months_total, go through the days of this month to find which day to use."""
        days = sorted(self.stats[month]['days'].keys())
        running_total = previous_months_total
        for d in days:
            total_files_in_day = self.stats[month]['days'][d]
            if idx < running_total + total_files_in_day:
                return d, running_total
            else:
                running_total += total_files_in_day
        # if we reached this point, it means that there are actually fewer files in the month than expected
        # by assuming each day has at least self.ass_num_files_per_day.

    def get_data_from_idx_month_and_day(self, idx, month, day, prev_days_total) -> Any:
        """
        From within the days' data, get the files.
        :param idx: the item index
        :param month: the selected month
        :param day: the selected day
        :param prev_days_total:
        :return:
        """
        day_folder = os.path.join(self.base_folder, month, day, '*.npy')
        files = glob.glob(day_folder)
        start = idx - prev_days_total
        try:
            assert start < len(files)-21
        except AssertionError as e:
            print("******PROBLEM:{} is not less than {}".format(start, len(files)-21))
            raise
        img_set = []
        for j in range(22):
            im = np.load(files[start +j])
            im = im[0]
            pic = np.zeros((256, 256))+im
            pic = np.expand_dims(pic, axis=2)
            img_set.append(pic)
        return np.array(img_set).astype(np.float32)

    def __getitem__(self, idx):
        """
        :param idx: the item index
        :return:
        """
        # Given item index, find the month.
        month, prev_months_total = self.get_month_from_idx(idx)

        # Given the item index and the total files from the previous month, find the day
        day, prev_days_total = self.get_day_from_idx_and_month(idx, month, prev_months_total)

        # Given the index, month, and day, retrieve the data.
        data = self.get_data_from_idx_month_and_day(idx, month, day, prev_days_total)

        images = data[:4]
        images_fixed = rearrange(images, 'seq h w c->seq c h w')
        target = data[4:]
        target_fixed = rearrange(target, 'seq h w c->seq c h w')
        return images_fixed, target_fixed


def test_radar_dataset():
    the_data = RadarDataset()
    size = the_data.__len__()
    # idx = random.randrange(0, size)
    idx = 8645
    inputs, targets = the_data.__getitem__(idx)
    assert inputs.shape == (4, 1, 256, 256)
    assert targets.shape == (18, 1, 256, 256)
    print("Success")


def main():
    test_radar_dataset()



if __name__ == "__main__":
    main()
