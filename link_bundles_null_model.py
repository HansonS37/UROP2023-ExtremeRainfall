import numpy as np
import scipy.stats as st
from sklearn.neighbors import KernelDensity

# get index of coordinates
def get_index(lat_t, lon_t):
    index = (180+lon_t)*400*4+(lat_t+50)*4
    return int(index)

# calculate spherical KDE
def shperical_kde(values, xy, bw_opt):
   # 求出KDE的曲线
   kde = KernelDensity(bandwidth=bw_opt, metric='haversine', kernel='gaussian', algorithm='ball_tree')
   kde.fit(values)
   datss = np.exp(kde.score_samples(xy))
   return datss

# number of events at corresponding position
events = np.load('num_ERE_final.npy')
events = events.T.reshape(1, 576000)[0]

# SCA links
SCA_links = np.load('SCA_links_final.npy')
links = SCA_links[:, 1]
selector = np.zeros(links.shape[0], dtype = 'int')
selector[:10000] = 1
selector = np.random.permutation(selector)
links = links[selector == 1]

# each time using this null model function will give one randomly generated output
def null_model(events, links):
    lat = np.arange(-50, 50, 0.25)
    lon = np.arange(-180, 180, 0.25)

    from itertools import product
    coords = np.array(list(product(lat, lon)))
    nol = links.shape[0]

    values = np.vstack([coords[links, 0], coords[links, 1]]).T
    values *= np.pi / 180.
    X, Y = np.meshgrid(lon, lat)
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy *= np.pi / 180.

    bw_opt = .2 * values.shape[0]**(-1./(2+4))

    noel2 = np.where(events > 2)[0]
    values = np.vstack([np.random.choice(coords[noel2, 0], nol), np.random.choice(coords[noel2, 1], nol)]).T
    values *= np.pi / 180.

    datss = shperical_kde(values, xy, bw_opt)
    return datss.reshape(X.shape)

# we use 600 times of this in our null model
from itertools import product
coords = np.array(list(product(lat, lon)))
nol = links.shape[0]
print(nol)

values = np.vstack([coords[links, 0], coords[links, 1]]).T
values *= np.pi / 180.
X, Y = np.meshgrid(lon, lat)
xy = np.vstack([Y.ravel(), X.ravel()]).T
xy *= np.pi / 180.

bw_opt = .2 * values.shape[0]**(-1./(2+4))
print(bw_opt)


dats = np.zeros((600, X.shape[0], X.shape[1]))
for i in range(600):
    noel2 = np.where(events > 2)[0]
    values = np.vstack([np.random.choice(coords[noel2, 0], nol), np.random.choice(coords[noel2, 1], nol)]).T
    values *= np.pi / 180.
    
    datss = shperical_kde(values, xy, bw_opt)
    dats[i] = datss.reshape(X.shape)
    print(i)

np.save('null_model_final.npy', dats)