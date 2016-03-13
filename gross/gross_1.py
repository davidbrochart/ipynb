try:
    import cPickle as pickle
except:
    import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from shapely.geometry import shape
from pandas import DataFrame
import numpy as np
import rasterio
import os
import requests
import zipfile
import scipy.ndimage

zippath = 'data/sa_dem_30s_grid.zip'
if not os.path.exists(zippath):
    r = requests.get('http://earlywarning.usgs.gov/hydrodata/sa_30s_zip_grid/sa_dem_30s_grid.zip')
    f = open(zippath, 'wb')
    f.write(r.content)
    f.close()

f = zipfile.ZipFile(zippath, 'r')
f.extractall('data/')
f.close()

zippath = 'data/sa_acc_30s_grid.zip'
if not os.path.exists(zippath):
    r = requests.get('http://earlywarning.usgs.gov/hydrodata/sa_30s_zip_grid/sa_acc_30s_grid.zip')
    f = open(zippath, 'wb')
    f.write(r.content)
    f.close()

f = zipfile.ZipFile(zippath, 'r')
f.extractall('data//')
f.close()

with rasterio.drivers():
    with rasterio.open('data/sa_dem_30s/sa_dem_30s/w001001.adf') as src:
        dem, = src.read()

with rasterio.drivers():
    with rasterio.open('data/sa_acc_30s/sa_acc_30s/w001001.adf') as src:
        acc, = src.read()

with open('data/partition.pkl', 'rb') as f:
    ws = pickle.load(f)

df_ws = DataFrame()
df_ws['odr'] = [str(i['order']).replace('[', '').replace(']', '').replace(' ', '') for i in ws]
df_ws['out'] = [i['outlet'] for i in ws]
df_ws['mask'] = [i['mask'] for i in ws]
df_ws['latlon'] = [i['latlon'] for i in ws]
df_ws = df_ws.set_index('odr')

lat0, lat1, lon0, lon1 = -21, 6, -80, -52 # bounding box in degrees
acc_res = 1 / 120 # flow accumulation resolution in degrees (0.008333)

def show_ws(groups, size):
    plt.close('all')
    fig = plt.figure(figsize = (size, size))
    ax = fig.add_subplot(111)
    m = Basemap(projection = 'cyl', resolution = 'l', area_thresh = 10000, llcrnrlat = lat0, urcrnrlat = lat1, llcrnrlon = lon0, urcrnrlon = lon1)
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(lat0, lat1, 10), labels=[1,1,0,0])
    m.drawmeridians(np.arange(lon0, lon1, 10), labels=[0,0,0,1])

    this_mask = np.zeros((int((lat1 - lat0) / acc_res), int((lon1 - lon0) / acc_res)), dtype = 'uint8')
    for this_group in groups:
        this_rand = int(np.random.uniform(1, 255))
        for this_odr in this_group:
            mask = df_ws.loc[this_odr, 'mask']
            latlon = df_ws.loc[this_odr,  'latlon']
            y0 = int(round((lat1 - latlon[0]) / acc_res))
            y1 = y0 + mask.shape[0]
            x0 = int(round((latlon[1] - lon0) / acc_res))
            x1 = x0 + mask.shape[1]
            this_mask[y0:y1, x0:x1] = this_mask[y0:y1, x0:x1] + mask * this_rand

    this_mask = np.where(this_mask == 0, np.nan, this_mask)
    m.imshow(this_mask, origin = 'upper', interpolation = 'nearest', zorder = 10, alpha = 1)

    plt.show()

def get_odr_startswith(odrs, odr):
    odr_list = []
    for this_odr in odrs:
        if this_odr.startswith(odr):
            odr_list.append(this_odr)
    return odr_list

def intersect_odr(odr1, odr2):
    odr = []
    for this_odr in odr1:
        if this_odr in odr2:
            odr.append(this_odr)
    return odr

def substract_odr(odr1, odr2):
    odr = []
    for this_odr in odr1:
        if this_odr not in odr2:
            odr.append(this_odr)
    return odr

def get_area(df_ws, odr):
    area = 0.
    for this_odr in odr:
        area += np.sum(df_ws.loc[this_odr, 'mask'])
    return area

def split_ws(odrs):
    odr0 = odrs
    area0 = get_area(df_ws, odr0)
    diff_area = np.inf
    for i in odr0:
        odr = get_odr_startswith(df_ws.index, i)
        area = get_area(df_ws, odr)
        diff = abs(area0 / 2 - area)
        if diff < diff_area:
            diff_area = diff
            odr1 = odr
    odr0 = substract_odr(odr0, odr1)
    return odr0, odr1
