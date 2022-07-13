"""
Here, we process raw *.uf data from PAGASA and convert it to *.np and store it in an organized system.
"""
import os
import glob
import pyart
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


source_folder = "/home/rolan/data1/Weather-Datasets/weather_data/weather_data/batch1/Radar/2019/Subic/subic_p/"
dest_folder_npy = "/home/rolan/data1/Weather-Datasets/npy-data/"
dest_folder_png = "/home/rolan/data1/Weather-Datasets/png-data/"
vmin = -30
vmax = 75
norm = plt.Normalize(vmin, vmax)
cmap = matplotlib.cm.get_cmap('jet')
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
if not os.path.exists(dest_folder_npy):
    os.mkdir(dest_folder_npy)

if not os.path.exists(dest_folder_png):
    os.mkdir(dest_folder_png)


def process_the_file(the_file):
    """
    Write the file as png and npy files in the appropriate folders.

    :param the_file: filename as /home/rolan/data1/Weather-Datasets/weather_data/weather_data/batch1/Radar/2019/Subic/subic_p/month/day/file_id.uf
    :return: None
    """
    f_arr = the_file.split('/')
    month = f_arr[12]
    day = f_arr[13]
    file_id = f_arr[14][:-3]
    """ Create the necessary folders"""
    dest_folder_month_npy = os.path.join(dest_folder_npy, str(month))
    dest_folder_day_npy = os.path.join(dest_folder_npy, str(month), str(day))
    dest_folder_month_png = os.path.join(dest_folder_png, str(month))
    dest_folder_day_png = os.path.join(dest_folder_png, str(month), str(day))
    if not os.path.exists(dest_folder_month_npy):
        os.mkdir(dest_folder_month_npy)
    if not os.path.exists(dest_folder_day_npy):
        os.mkdir(dest_folder_day_npy)
    if not os.path.exists(dest_folder_month_png):
        os.mkdir(dest_folder_month_png)
    if not os.path.exists(dest_folder_day_png):
        os.mkdir(dest_folder_day_png)

    dest_file_npy = os.path.join(dest_folder_day_npy, file_id + '.npy')
    dest_file_png = os.path.join(dest_folder_day_png, file_id + '.png')

    """Process the file and write the npy and png"""
    try:
        radar = pyart.io.read_uf(the_file)
        radar = radar.extract_sweeps([0])
        grid = pyart.map.grid_from_radars(radar, grid_shape=(1, 256, 256), grid_limits=((0., 20000,), (-240000., 240000.), (-240000, 240000.)))
        grid = grid.fields['corrected_reflectivity']['data'].copy()
        grid = np.array(grid)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = fig.colorbar(sm)
        cbar.ax.set_title("dbz")
        ax.set_title(id)
        ax.imshow(grid[0], cmap='jet', vmin=vmin, vmax=vmax)
        plt.savefig(dest_file_png)
        np.save(dest_file_npy, grid)
        print(dest_file_npy)
        plt.close(fig)
    except Exception as ex:
        print(str(ex))


def main():
    the_months = os.path.join(source_folder, "*")
    month_folders = glob.glob(the_months)
    print("There are {} months.".format(len(month_folders)))
    for the_month in month_folders:
        mo = the_month.split('/')[-1]
        day_folders = glob.glob(os.path.join(the_month, "*"))
        ndays = len(day_folders)
        print("For month: {}, there are {} days".format(mo, ndays))
        for day in day_folders:
            day_files = glob.glob(os.path.join(day, "*.uf"))
            for file in day_files:
                process_the_file(file)


if __name__ == "__main__":
    main()
