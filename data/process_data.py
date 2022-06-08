import numpy as np
import pyart as pyart
import cv2
import matplotlib.pyplot as plt
import glob as glob
import os
import matplotlib

files = glob.glob('/weather_data/batch1/Radar/2019/Subic/subic_p/*/*/*.uf')
vmin = -30
vmax = 75
norm = plt.Normalize(vmin, vmax)
cmap = matplotlib.cm.get_cmap('jet')
sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = cmap)
sm.set_array([])
"""
if not os.path.exists('./data_full/'):
    os.mkdir('./data_full/')
for f in files:
    fs = f.split('/')
    mos = fs[7]
    day = fs[8]
    id = fs[9][:-3]
    d = './data_full/' + str(mos) + '/' + str(day) + '/'
    if not os.path.exists('./data_full/' + str(mos) + '/'):
        os.mkdir('./data_full/' + str(mos) + '/')
    if not os.path.exists('./data_full/' + str(mos) + '/' + str(day) + '/'):
        os.mkdir('./data_full/' + str(mos) + '/' + str(day) + '/')
    if not os.path.exists(d + str(id) + '.png') or not os.path.exists(d + str(id) + '.npy'):
"""
data_root = "~/data1/Weather-Datasets/weather_data/weather_data/batch1/Radar/2019/Subic/subic_p"
dest_root = "/home/rolan/data1/Weather-Datasets/processed/subic_p"

id= "052006"
a_file = os.path.join(data_root, "01", "09", "20190109" + id + ".uf")
dest_file =  os.path.join(dest_root, "01", "09")
try:
    radar = pyart.io.read_uf(a_file)
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
    plt.savefig(dest_file + '/' + str(id) + '.png')
    np.save(dest_file + '/' + str(id) + '.npy', grid)
    print(dest_file + '/' + str(id) + '.npy')
except:
    pass
im = np.load(dest_file + '/' + str(id) + '.npy')
print("Shape is:{}".format(im.shape))