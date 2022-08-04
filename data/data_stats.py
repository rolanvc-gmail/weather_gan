"""
We gather stats from the data. Stats like num months available, num days for each month, num files for each day, etc.
Also we want to be able to create a batch by retrieving a particular image-sequence from an index.
Each day has a different number of files, n. Each day can have up to (n-22) image-sequences, starting from idx=0 to idx=n-22.
We need to assume each day as an N number of sequences to be able to reference a specific sequence.  If a day has more than N sequences, we discard those sequences.
If a day has less, we discard that day. Therefore, we need to find the N that maximizes the data image-sequences we can index.
"""
import os
import glob
from process_data import dest_folder_npy
from typing import Tuple
import pickle
base_folder_npy = dest_folder_npy


def gather_stats(base_folder) -> dict:
    """"
    Given base_folder, we asssume the files are in base_folder/mo/day/*.uf.
    We count the months and days for each month, and number of files for each day and return this as a dictionary:
    {mo1: {'tot_days':N1, 'days':{ '01':n11, '02':n12...},
     mo2: {'tot_days':N2, 'days':{ '01':n21, '02':n22...},
     }
     }
    """
    datafile = os.path.join(base_folder_npy, "meta.pickle")
    data_dict = {}
    """
    if the pickle file exists, read from that; else, we compute data_dict and save it to pickle file.
    """
    if os.path.exists(datafile):
        with open(datafile, 'rb') as the_datafile:
            data_dict = pickle.load(the_datafile)

    else:
        files = glob.glob(os.path.join(base_folder, "*"))
        print("there are {} months.".format(len(files)))
        for mo in files:
            mo_idx = mo.split('/')[-1]
            days = glob.glob(os.path.join(mo, "*"))
            data_dict[mo_idx] = {"tot_days": len(days), "days": {}}
            print("for month {}, there are {} days.".format(mo_idx, len(days)))
            for day in days:
                day_idx = day.split('/')[-1]
                files_path = os.path.join(day, "*")
                files = glob.glob(files_path)
                """
                a data point is 22 files. so if a day has 22 files, it only has 1 data point. so, in general, a day with N files has N-21 data points.
                """
                data_dict[mo_idx]["days"][day_idx] = len(files) - 21
                print("for month:{}, day:{}, there are {} files".format(mo_idx, day_idx, len(files)))
        with open(datafile, "wb") as the_datafile:
            pickle.dump(data_dict, the_datafile, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dict


"""
A problem with the dataset is that each day has a different number of data files. To make indexing the data easier, we compute a number of a files that we
will assume each day, except a few will have.  If this assumed number is high, more days will be excluded, reducing our total data size. If this 
number is too low, we also reduce our data set size since we assume total size is num_months * num_days * files per day.
"""


def get_min_files_in_days(data, cutoff) -> Tuple[int, int, int, int]:
    """
    Given the data_dict computed above, and a cutoff, we compute:
    1. total_days:int, the total days of the dataset.
    2. cutoff:int, the cutoff passed as a parameter.
    3. min_files:int, the smallest number of files (greater than the cutoff) that a day can have.
    4. lower_than_cutoff:int, an array of days in mo/day format which has less files than the cutoff.
    :param data:
    :param cutoff:
    :return:
    """
    min_files = 1000
    lower_than_cutoff = []
    total_days = 0
    months = data.keys()
    for mo in months:  # iterate through the months
        days = data[mo]['days'].keys()
        for day in days:  # iterate through the days in the month
            total_days += 1
            n_files = data[mo]['days'][day]
            if n_files < cutoff:
                lower_than_cutoff.append("{}/{}".format(mo, day))
                continue
            if n_files < min_files:
                min_files = n_files

    return total_days, cutoff, min_files, len(lower_than_cutoff)


def check_cutoff(cutoff) -> Tuple[int, int, int]:
    """
    Given a cutoff, we estimate the total data size we expect to have.
    
    :param cutoff: the cutoff (assumed num of files in a day)
    :return: [int,int] the total number of days, assumed number of files per day, est total data points
    """
    data = gather_stats(base_folder_npy)
    total_days, cutoff, min_num_files, lower_than_cutoff = get_min_files_in_days(data, cutoff)
    print("\n******\n For Cutoff:{}".format(cutoff))

    print("Total days:{}".format(total_days-lower_than_cutoff))
    print("smallest number of files a day has (greater than cutoff):{}".format(min_num_files))
    print("There are {} days with less than {} files.".format(lower_than_cutoff, cutoff ))
    total_data_points = (total_days - lower_than_cutoff) * min_num_files
    print("Estimated Total Data Points is: {}".format(total_data_points))
    return total_days, min_num_files, total_data_points


def main():
    """
    determine the best cutoff that yields the largest dataset size...
    :return:
    """
    stats = {}
    for cutoff in range(50, 150):
        tot_num_days, best_num_files_per_day, est_data_size = check_cutoff(cutoff)
        stats[cutoff] = est_data_size
    best = max(stats, key=stats.get)
    print()
    print()
    print("Best cutoff is:{} with estimated data size of {}".format(best, stats[best]))


if __name__ == "__main__":
    main()

