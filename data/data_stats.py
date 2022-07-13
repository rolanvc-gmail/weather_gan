"""
We gather stats from the data. Stats like num months available, num days for each month, num files for each day, etc.
"""
import os
import glob
from process_data import dest_folder_npy
base_folder_npy = dest_folder_npy




def compute_stats():
    data_dict = {}
    files = glob.glob(os.path.join(base_folder_npy, "*"))
    print("there are {} months.".format(len(files)))
    for mo in files:
        mo_idx = mo.split('/')[-1]
        days = glob.glob(os.path.join(mo, "*"))
        data_dict[mo_idx] = { "tot_days": len(days), "days": {}}
        print("for month {}, there are {} days.".format(mo_idx, len(days)))
        for day in days:
            day_idx = day.split('/')[-1]
            files_path = os.path.join(day, "*")
            files = glob.glob(files_path)
            data_dict[mo_idx]["days"][day_idx] = len(files)
            print("for month:{}, day:{}, there are {} files".format(mo_idx, day_idx, len(files)))
    return data_dict


def main():
    data = compute_stats()
    print(data)


if __name__ == "__main__":
    main()

