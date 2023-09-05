'''
Calculate the ES between two nodes, using Numba and parallel computing. 
Save data as (i,j,k) for ES value that are larger than threshold, 
where i, j represent two nodes, and k represents the ES value. 
'''

import numpy as np
import pickle
from collections import defaultdict
from numba import jit
import time
import multiprocessing as mp

# import ERE info of all places
with open("ERE_start_days", "rb") as fp:   # Unpickling
    start_date_ERE = pickle.load(fp)

# import threshold 
sig_level = np.load("trmm7_P_2000_3ms_mnoe78_thresholds_005_tm10_2.npy")

# numba
@jit(nopython=True)
def calc_ES_ij(day_list_i, day_list_j):
    ES_ij = 0
    len_i, len_j = len(day_list_i), len(day_list_j)
    i, j = 0, 0
    while i < len_i and j < len_j:
        threshold_list_i = [day_list_i[i+1]-day_list_i[i] if i < len_i-1 else np.inf, day_list_i[i]-day_list_i[i-1] if i > 0 else np.inf]
        threshold_list_j = [day_list_j[j+1]-day_list_j[j] if j < len_j-1 else np.inf, day_list_j[j]-day_list_j[j-1] if j > 0 else np.inf]
        tau_1 = min(min(threshold_list_i), min(threshold_list_j)) / 2
        tau = min(tau_1, 10)  # tau_2=10
        if abs(day_list_i[i] - day_list_j[j]) < tau:
            ES_ij += 1
            i += 1
            j += 1
        elif day_list_i[i] < day_list_j[j]:
            i += 1
        else:
            j += 1
    if ES_ij >= sig_level[max(len_i, len_j)][min(len_i, len_j)]:
        return ES_ij
    else:
        return 0


# Define function for parallel
def ES_p(name, index):
    start_index = index[0]
    end_index = index[1]
    ES_part = []
    for i in range(start_index, end_index):  # need to specify
        for j in range(i+1, 400*1440):
            row_i, col_i = divmod(i, 400)  # find the place of node i in matrix
            row_j, col_j = divmod(j, 400)
            day_list_i = start_date_ERE[row_i][col_i]
            day_list_j = start_date_ERE[row_j][col_j]
            ES_ij = 0
            if isinstance(day_list_i, list) and isinstance(day_list_j, list):
                ES_ij = calc_ES_ij(day_list_i, day_list_j)
            if ES_ij > 0:
                ES_part.append((i, j, ES_ij))
    return ES_part


a = time.time()

if __name__ == '__main__':
    num_cores = int(mp.cpu_count())
    print("Computer has: " + str(num_cores) + "cpus")
    pool = mp.Pool(32)  # specify at 30/32 cpus
    para_dict = {}
    for i in range(32):  # specify number of cpu
        para_dict[i] = [500000 + 2375*i, 500000 + 2375 * (i+1)]  
        # modify according to the indices of nodes that u need to calculate
    results = [pool.apply_async(ES_p, args=(name, index)) for name, index in para_dict.items()]
    pool.close()
    results = [p.get() for p in results]

b = time.time()

print(b-a)
print("success")

with open("whole500000_576000", "wb") as fp:  # save file
    pickle.dump(results, fp)

print("file success")
