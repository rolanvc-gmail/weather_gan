import random
import os
import glob
import numpy as np
from torch.utils.data.dataset import Dataset
from data.data_stats import gather_stats, check_cutoff
from process_data import dest_folder_npy
from typing import Any, Tuple

n_files_in_day = 116  # from data_stats computation.


def get_start_from_idx_month_day(idx, month, day, prev_total):
    start = idx - prev_total
    return start


class RadarDataset(Dataset):
    def __init__(self):
        self.base_folder = dest_folder_npy
        self.stats = gather_stats(self.base_folder)

        stats_size = {}
        stats_all = {}
        for cutoff in range(50, 150):
            total_days, ass_num_files, est_data_size = check_cutoff(cutoff)
            stats_size[cutoff] = est_data_size
            stats_all[cutoff] = {'total_days': total_days, 'ass_num_files': ass_num_files, 'est_size': est_data_size}
            self.best_cutoff = max(stats_size, key=lambda o: stats_size[o])

        self.months = sorted(self.stats.keys())
        self.data_size = stats_all[self.best_cutoff]['est_size']
        self.ass_num_files_per_day = stats_all[self.best_cutoff]['ass_num_files']

    def __len__(self):
        return self.data_size

    def get_month_from_idx(self, idx):
        """
        get which month the idx most likely falls in.
        :param idx:
        :return:
        """
        running_total = 0
        for m in self.months:
            if idx < running_total + self.stats[m]['tot_days'] * self.ass_num_files_per_day:
                return m, running_total
            else:
                running_total += self.stats[m]['tot_days'] * self.ass_num_files_per_day

    def get_day_from_idx_and_month(self, idx: int, month: int, previous_months_total: int) -> Tuple[Any, int]:
        days = sorted(self.stats[month]['days'].keys())
        running_total = previous_months_total
        for d in days:
            if idx < running_total + self.stats[month]['days'][d]:
                return d, running_total
            else:
                running_total += self.stats[month]['days'][d]

    def get_data_from_idx_month_and_day(self, idx, month, day, prev_days_total) -> Any:
        """
        From within the days' data, get the files.
        :param idx:
        :param month:
        :param day:
        :param prev_days_total:
        :return:
        """
        day_folder = os.path.join(self.base_folder, month, day, '*.npy')
        files = glob.glob(day_folder)
        start = get_start_from_idx_month_day(idx, month, day, prev_days_total)
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

        :param idx:
        :return: a np.array of size 22 x 256 x 256 x 1
        """
        month, prev_months_total = self.get_month_from_idx(idx)
        day, prev_days_total = self.get_day_from_idx_and_month(idx, month, prev_months_total)
        data = self.get_data_from_idx_month_and_day(idx, month, day, prev_days_total)
        images = data[:4]
        target = data[4:]
        return images, target


def test_radar_dataset():
    the_data = RadarDataset()
    size = the_data.__len__()
    idx = random.randrange(0, size)
    data = the_data.__getitem__(idx)
    assert data.shape == (22, 256, 256, 1)


def main():
    test_radar_dataset()



if __name__ == "__main__":
    main()
