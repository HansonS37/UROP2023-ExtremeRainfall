# fig 3
# import all possibly necessary packages
import numpy as np
from sklearn.neighbors import KernelDensity
from itertools import product
import scipy.stats as st
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import csv
import multiprocessing
import matplotlib.cm as cm
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import more_itertools as mit
from scipy.stats import gaussian_kde

# get the index of a position, for 400*1440 matrix version
def get_index(lat_t, lon_t):
    index = (180+lon_t)*400*4+(lat_t+50)*4
    return int(index)

# get the index of a position, for 1440*400 matrix version
def get_index_new(lat_t, lon_t):
    index = (lat_t+50)*1440*4 + (180+lon_t)*4
    return int(index)

# the geographic_link_dist_ref_sn in author's code
from math import sin, cos, acos, asin, sqrt
import random
import time

def geographic_link_dist_ref_sn(lat, lon, lat_t, lon_t, noe, frac):

    # idx = int(get_index(lat_t, lon_t))
    idx = int(get_index_new(lat_t, lon_t))
    la = len(lat)
    lo = len(lon)
    n = la * lo

    lat_seq = np.repeat(lat, lo)
    lon_seq = np.tile(lon, la)
    sin_lon = np.sin(lon_seq * np.pi / 180)
    cos_lon = np.cos(lon_seq * np.pi / 180)
    sin_lat = np.sin(lat_seq * np.pi / 180)
    cos_lat = np.cos(lat_seq * np.pi / 180)

    rn = np.where(noe > 2)[0].shape[0]
    ang_dist = np.zeros(int(2 * rn * frac), dtype = 'uint16')

    random.seed(int(time.time()))
    c = 0

    for j in range(n):
        if noe[idx] > 2 and noe[j] > 2:
            r = random.random()
            if r < frac:
                expr = sin_lat[idx] * sin_lat[j] + cos_lat[idx] * cos_lat[j] * (sin_lon[idx] * sin_lon[j] + cos_lon[idx] * cos_lon[j])
                if expr > 1:
                    expr = 1.0
                elif expr < -1:
                    expr = -1.0
                ang_dist[c] = int(acos(expr) * 6371)
                c += 1

    return ang_dist[ang_dist != 0]


# define parameters
x_min = 100
lat = np.arange(-50, 50, 0.25)
lon = np.arange(-180, 180, 0.25)

la = len(lat)
lo = len(lon)
nodes = la * lo
m = nodes * (nodes - 1) / 2

perc = 95
i = 0
tm = 10

# change the 400*1440 version index to 1440*400 version index
def transfer_index(index):
    lat = index%400
    lon = index//400
    new_index = lat*1440 + lon
    return new_index

# num_ERE.npy is the array containing the number of EREs in a position
noe = np.load('num_ERE_final.npy')
noe = noe.flatten()
print(noe)

### change the index to 1440*400 matrix version
index_seq = np.arange(0,576000)
print(index_seq)
index_seq = transfer_index(index_seq)
print(index_seq)
print(len(index_seq) == len(np.unique(index_seq)))
print(index_seq)

### change the noe according to index_seq
noe = noe[index_seq]
print(noe)

from math import sin, cos, sqrt, atan2, radians

# Haversine distance calculation
def Hav_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers (mean radius = 6,371 km)
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

# get the position of an index, 400*1440 version
def pos(coord):
    lonindex = coord // 400
    latindex = coord % 400
    lat = latindex*0.25 - 50
    lon = lonindex*0.25 - 180
    return lat, lon

### get the position of an index, 1440*400 version
def pos_new(coord):
    latindex = coord // 1440
    lonindex = coord % 1440
    lat = latindex*0.25 - 50
    lon = lonindex*0.25 - 180
    return lat, lon

# read the links from SCA to world
links_SCA_world_ES = np.load('SCA_links_final.npy')
### find links
links = np.array([], dtype='int')
gdists = np.array([])
for i in range(len(links_SCA_world_ES)):
    link = links_SCA_world_ES[i]
    links = np.concatenate((links, [link[1]]))
    lat1, lon1 = pos_new(link[0])
    lat2, lon2 = pos_new(link[1])
    distance = Hav_distance(28, 79, lat2, lon2)
    gdists = np.concatenate((gdists, [distance]))
# randomly take 10000 points
selector = np.zeros(links.shape[0], dtype = 'int')
selector[:10000] = 1
selector = np.random.permutation(selector)
links = links[selector == 1]
gdists = gdists[selector == 1]

# get index array according to central 25 points
index = np.array([], dtype = 'int')

for lat_t in [27.5, 27.75, 28., 28.25, 28.5]:
    for lon_t in [78.5, 78.75, 79., 79.25, 79.5]:
        index = np.concatenate((index, [get_index_new(lat_t, lon_t)]))

print(index)
# index = transfer_index(index)
index = np.unique(index)

# angular distance
ang_dist = gdists.copy()
lat_t = 28
lon_t = 79
ang_dist_ref = geographic_link_dist_ref_sn(lat, lon, lat_t, lon_t, noe, .1)
ang_dist_ref = ang_dist_ref[ang_dist_ref > x_min]

logbins_reg = np.logspace(np.log10(ang_dist_ref.min()), np.log10(ang_dist_ref.max()), 30)
loghist_ref_reg = np.histogram(ang_dist_ref, bins = logbins_reg, density = True)
logx_reg = logbins_reg[:-1] + (logbins_reg[1:] - logbins_reg[:-1]) / 2.
loghist_reg = np.histogram(ang_dist, bins = logbins_reg, density = True)

import time
# KDE result of angular distance
start = time.time()
kernel = gaussian_kde(ang_dist_ref)
end =  time.time()

# log the x axis
start = time.time()
kde_ref = kernel.evaluate(logx_reg)
kde_reg = kernel.evaluate(logx_reg)
tkde_reg = kde_reg[logx_reg > x_min]
kernel = kernel.evaluate(ang_dist)
end =  time.time()

# to store all logx with greater than x_min
tlogx_reg = logx_reg
tkde_reg = kde_reg[logx_reg > x_min]
tlogy_reg = loghist_reg[0]
tlogy_ref_reg = loghist_ref_reg[0][logx_reg > x_min]

# get the grid of coordinates
from itertools import product
coords = np.array(list(product(lat, lon)))
lat0 = coords[index, 0]
lon0 = coords[index, 1]
lat1 = coords[links, 0]
lon1 = coords[links, 1]
nol = links.shape[0]

# define short and long links
links_short = links[gdists < 2500.]
links_long = links[gdists > 2500.]

# angular distance
ang_dist_short = ang_dist[ang_dist < 2500.]
ang_dist_long = ang_dist[ang_dist >= 2500.]

# find short and long links based on coords
lat1_short = coords[links_short, 0]
lon1_short = coords[links_short, 1]
lat1_long = coords[links_long, 0]
lon1_long = coords[links_long, 1]

# read null model data
dats = np.load('null_model_new.npy')
mean = np.mean(dats, axis = 0)
std = np.std(dats, axis = 0)
perc90 = st.scoreatpercentile(dats, 90, axis = 0)
perc95 = st.scoreatpercentile(dats, 95, axis = 0)
perc99 = st.scoreatpercentile(dats, 99, axis = 0)
perc995 = st.scoreatpercentile(dats, 99.5, axis = 0)
perc999 = st.scoreatpercentile(dats, 99.9, axis = 0)

# KDE calculation for links
values = np.vstack([coords[links, 0], coords[links, 1]]).T
values *= np.pi / 180.

X, Y = np.meshgrid(lon, lat)
xy = np.vstack([Y.ravel(), X.ravel()]).T
xy *= np.pi / 180.

bw_opt = .2 * values.shape[0]**(-1./(2+4))
kde = KernelDensity(bandwidth=bw_opt, metric='haversine', kernel='gaussian', algorithm='ball_tree')
kde.fit(values)

dat = np.exp(kde.score_samples(xy))
dat = dat.reshape(X.shape)

# set significant level to 99.9 and get corresponding variables needed
sig = perc999
fac = np.intersect1d(np.where(dat.flatten() > sig.flatten())[0], links)
ix = np.in1d(links.ravel(), fac).reshape(links.shape)
lat1 = coords[links[ix], 0]
lon1 = coords[links[ix], 1]
lat1_uc = coords[links, 0]
lon1_uc = coords[links, 1]
area = np.ones((len(lat), len(lon))) * -1
area[dat>sig] = 10
area[0] = -1
area[-1] = -1
area_noe = np.ones((len(lat), len(lon)))
area_noe[noe.reshape((la, lo)) >= 3] = -1

# area fraction of significant links and fraction of significant links
fac_short = np.intersect1d(np.where(dat.flatten() > sig.flatten())[0], links_short)
ix_short = np.in1d(links_short.ravel(), fac_short).reshape(links_short.shape)
lat1_short = coords[links_short[ix_short], 0]
lon1_short = coords[links_short[ix_short], 1]

fac_long = np.intersect1d(np.where(dat.flatten() > sig.flatten())[0], links_long)
ix_long = np.in1d(links_long.ravel(), fac).reshape(links_long.shape)
lat1_long = coords[links_long[ix_long], 0]
lon1_long = coords[links_long[ix_long], 1]

print("area fraction of significant links = ", dat.flatten()[dat.flatten() > sig.flatten()].shape[0] / float(dat.flatten().shape[0]))
probs = .01 * np.ones_like(links)
probs = probs[ix]
probs[0] = .06
print("fraction of significant links = ", links[ix].shape[0] / float(links.shape[0]))

# plot 3a
datnan = np.ones((len(lat), len(lon))) * -1
# Define the projection
fig, ax = plt.subplots(1, 1, figsize=(30, 18),subplot_kw={'projection': ccrs.PlateCarree()})
# Define your longitude and latitude values
lat = np.linspace(-50, 50, 400)
lon = np.linspace(-180, 180, 1440)
lon_grid, lat_grid = np.meshgrid(lon, lat)
# Contour the data (modify accordingly)
cs = ax.contourf(lon_grid, lat_grid, datnan, transform=ccrs.PlateCarree(), cmap=plt.cm.Blues, alpha=0.5, extend='both')
# Add coastlines
ax.coastlines()
# Add meridians and parallels
ax.gridlines(xlocs=np.arange(0, 360, 40), ylocs=np.arange(-50, 75, 25), color='gray', linewidth=0.2)
# great circle paths to draw
gc = ccrs.Geodetic()
gc_lon0 = lon0
gc_lat0 = lat0
gc_lon1 = lon1_uc
gc_lat1 = lat1_uc
for i in range(10000):
    ax.plot([79, gc_lon1[i]], [28, gc_lat1[i]], transform=ccrs.PlateCarree(), color='blue', linewidth=0.1)
    print(i)
ax.set_global()
# Add other features as needed
# ax.add_feature(cfeature.LAND, facecolor='lightgray')
plt.show()

# dotted distribution of endpoints of links 3a
datnan = np.ones((len(lat), len(lon))) * -1
# Define the projection
fig, ax = plt.subplots(1, 1, figsize=(30, 18),subplot_kw={'projection': ccrs.PlateCarree()})
# Add coastlines
ax.coastlines()
# Add meridians and parallels
ax.gridlines(xlocs=np.arange(0, 360, 40), ylocs=np.arange(-50, 75, 25), color='gray', linewidth=0.2)
# If you have great circle paths to draw
gc = ccrs.Geodetic()
gc_lon0 = lon0
gc_lat0 = lat0
gc_lon1 = lon1_uc
gc_lat1 = lat1_uc
ax.scatter(gc_lon1, gc_lat1, transform=ccrs.PlateCarree(), color='blue', linewidth=0.1)
ax.set_global()
# Add other features as needed
# ax.add_feature(cfeature.LAND, facecolor='lightgray')
plt.show()

# get significant level parameters for link bundles(3b)
sigdat = dat.copy()
sigdat[dat > mean + 5 * std] = 5.5
sigdat[dat <= mean + 5 * std] = 4.5
sigdat[dat <= mean + 4 * std] = 3.5
sigdat[dat <= mean + 3 * std] = 2.5
sigdat[dat <= mean + 2 * std] = 1.5
sigdat[-1] = 1.5
sigdat[0] = 1.5
conts = [2., 3., 4., 5.]
gc_lat0 = lat0
gc_lon0 = lat0
gc_lon1 = lon1_long
gc_lat1 = lat1_long
gc_lon2 = lon1_short
gc_lat2 = lat1_short

# plot all link bundles, 3b
fig, ax = plt.subplots(1, 1, figsize=(30, 18),subplot_kw={'projection': ccrs.PlateCarree()})
# longer than 2500km link bundles
for i in range(len(gc_lon1)):
    ax.plot([79, gc_lon1[i]], [28, gc_lat1[i]], transform=ccrs.PlateCarree(), color='blue', linewidth=0.2)
# shortter than 2500km link bundles
for i in range(len(gc_lon2)):
    ax.plot([79, gc_lon2[i]], [28, gc_lat2[i]], transform=ccrs.PlateCarree(), color='red', linewidth=0.2)
# plot other features
ax.coastlines()
ax.set_global()
ax.gridlines(draw_labels=True)
plt.show()

# plot the contour map of 3b
fig, ax = plt.subplots(1, 1, figsize=(30, 18),subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.coastlines()
ax.gridlines(draw_labels=True)

lat = np.linspace(-50, 50, 400)
lon = np.linspace(-180, 180, 1440)
lon_grid, lat_grid = np.meshgrid(lon, lat)

cs = ax.contourf(lon_grid, lat_grid, sigdat, transform=ccrs.PlateCarree(), alpha=0.5)
plt.show()

# plot contour for standard deviation of null model
mdats = std
fig, ax = plt.subplots(1, 1, figsize=(30, 18),subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_global()
ax.coastlines()
ax.gridlines(draw_labels=True)

lat = np.linspace(-50, 50, 400)
lon = np.linspace(-180, 180, 1440)
lon_grid, lat_grid = np.meshgrid(lon, lat)

contour_levels = np.linspace(mdats.min(), mdats.max(), 11)
cs = ax.contourf(lon_grid, lat_grid, mdats, contour_levels, transform=ccrs.PlateCarree(), alpha=0.5)

plt.show()

# plot contour for mean of null model
mdats = mean
fig, ax = plt.subplots(1, 1, figsize=(30, 18),subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_global()
ax.coastlines()
ax.gridlines(draw_labels=True)

lat = np.linspace(-50, 50, 400)
lon = np.linspace(-180, 180, 1440)
lon_grid, lat_grid = np.meshgrid(lon, lat)

contour_levels = np.linspace(mdats.min(), mdats.max(), 11)
cs = ax.contourf(lon_grid, lat_grid, mdats, contour_levels, transform=ccrs.PlateCarree(), alpha=0.5)

plt.show()