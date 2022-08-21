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
import statistics


def gather_stats(base_folder) -> dict:
    """"
    Given base_folder, we asssume the files are in base_folder/mo/day/*.uf.
    We count the months and days for each month, and number of files for each day and return this as a dictionary:
    {mo1: {'tot_days':N1, 'total_datapoints':NN1, 'days':{ '01':n11, '02':n12...},
     mo2: {'tot_days':N2, 'total_datapoints':NN2, 'days':{ '01':n21, '02':n22...},
     }
     }
    """
    datafile = os.path.join(base_folder_npy, "meta_data.pickle")
    data_dict = {}
    """
    if the pickle file exists, read from that; else, we compute data_dict and save it to pickle file.
    """
    if os.path.exists(datafile):
        with open(datafile, 'rb') as the_datafile:
            data_dict = pickle.load(the_datafile)
            print("Reading metadata from {}".format(datafile))

    else:  # the file does not exist, build the file.
        months = glob.glob(os.path.join(base_folder, "*"))  # get the list of months inside base_folder
        # print("there are {} months.".format(len(files)))
        for mo in months:
            mo_idx = mo.split('/')[-1]  # mo_idx is '1', '2',... '12'
            days = glob.glob(os.path.join(mo, "*"))  # get list of days inside months' folder: base_folder/mo
            data_dict[mo_idx] = {"tot_days": len(days), "days": {}}  # record the total number of days for that month
            for day in days:
                day_idx = day.split('/')[-1]
                files_path = os.path.join(day, "*")
                files = glob.glob(files_path)
                """
                a data point is 22 files. so if a day has 22 files, it only has 1 data point. so, in general, a day with N files has N-21 data points.
                """
                data_dict[mo_idx]["days"][day_idx] = len(files) - 21
            total_datapoints_mo = 0
            data_dict[mo_idx]["total_datapoints"] = 0
            for d in data_dict[mo_idx]["days"].keys():
                total_datapoints_mo += data_dict[mo_idx]["days"][d]
            data_dict[mo_idx]["total_datapoints"] += total_datapoints_mo

        total_datapoints_all = 0
        data_dict["total_datapoints_all"] = total_datapoints_all
        for mo in data_dict.keys():
            if mo != "total_datapoints_all":
                data_dict["total_datapoints_all"] += data_dict[mo]["total_datapoints"]

        with open(datafile, "wb") as the_datafile:
            pickle.dump(data_dict, the_datafile, protocol=pickle.HIGHEST_PROTOCOL)
            print("Writing metadata to {}".format(datafile))

    return data_dict


def main():
    data_dict = gather_stats(base_folder_npy)
    months = data_dict.keys()
    print("Data has {} months".format(len(months)))
    num_datapoints = []
    for m in months:
        if m != "total_datapoints_all":
            num_datapoints.append(data_dict[m]["total_datapoints"])

    total_datapoints_all = sum(num_datapoints)
    mean_datapoints = statistics.mean(num_datapoints)
    print("there are {} total datapoints. with an average of {} per month".format(total_datapoints_all, mean_datapoints))


if __name__ == "__main__":
    main()

