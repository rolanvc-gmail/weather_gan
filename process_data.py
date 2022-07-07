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
source_files = os.path.join(source_folder, "*/*/*.uf")
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

def main():
    files = glob.glob(source_files)
    for the_file in files:
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
        """Process each file."""
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
        except Exception as ex:
            print(str(ex))


if __name__ == "__main__":
    main()
