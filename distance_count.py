'''
Count distance distribution given the list of pair of nodes.
Use parallel computing to enhance the speed.
'''

from math import sin, cos, asin, sqrt, pi
import numpy as np
import pickle
import multiprocessing as mp


# define the function get lat and lon
def pos(coord):
    lonindex = coord // 400
    latindex = coord % 400
    lat = (latindex*0.25 - 50) * pi / 180
    lon = (lonindex*0.25 - 180) * pi / 180
    return lat, lon


# define the function to calculate distance
def Hav_distance(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon_orig = lon2 - lon1
    dlon = min(dlon_orig, 2*pi - dlon_orig)
    a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon/2) ** 2)
    return 2*6371*asin(sqrt(a))


# read data
with open("SCA&Whole_final.pkl", "rb") as fp:  # this is output of event_sync.py
    data = pickle.load(fp)

# define the initial count value for different ranges
x1 = np.arange(10, 100, 10)
x2 = np.arange(100, 200, 25)
x22 = np.arange(200, 1000, 100)
x3 = np.arange(1000, 2000, 250)
x4 = np.arange(2000, 10000, 1000)
x5 = np.arange(10000, 50000, 2500)
x = np.concatenate((x1, x2, x22, x3, x4, x5))  # x is the range of distances

count_list = [0 for i in range(len(x)+1)]
not_in_range = 0

# find the length of the data and use loop to count
datalen = len(data)


# Calculate distance and add to the category
def distance_cal(name, data_part):
    count_list_part = [0 for i in range(len(x))]
    not_in_range_part = 0
    for tup in data_part:
        point1 = tup[0]
        point2 = tup[1]
        n = tup[2]  # n is the ES value between these two nodes
        lat1, lon1 = pos(point1)
        lat2, lon2 = pos(point2)
        distance = Hav_distance(lat1, lon1, lat2, lon2)
        if distance <= 10:
            not_in_range_part += n  # note that we add n, instead of 1
        for j in range(len(x)):
            if distance < x[j]:
                count_list_part[j-1] += n
                break
    count_list_part.append(not_in_range_part)
    return count_list_part


if __name__ == '__main__':
    num_cores = int(mp.cpu_count())
    print("computer has: " + str(num_cores) + "cpus")
    pool = mp.Pool(datalen)  # specify num of cpu
    para_dict = {}
    for i in range(datalen):  # specify num of cpu
        para_dict[i] = data[i]
    results = [pool.apply_async(distance_cal, args=(name, index)) for name, index in para_dict.items()]
    pool.close()
    results = [p.get() for p in results]

# combine the results from parallel computing together
for i in range(len(results)):
    for j in range(len(results[i])):
        count_list[j] += results[i][j]

with open("count_list_finalSCA", "wb") as fp:
    pickle.dump(count_list, fp)
