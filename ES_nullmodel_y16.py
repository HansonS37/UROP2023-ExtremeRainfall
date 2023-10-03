import numpy as np
import scipy.stats as st
import pickle



def calc_ES_ij(day_list_i, day_list_j):
    # Input: two lists/arrays of ERE dates
    # Output: number of ES if ES between them is higher than the threshold; 0 otherwise (not attain the threshold)
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
    return ES_ij



### generate all dates
tlen = 5844 #total number of days
nodes = 2 #number of nodes considered
y = 16 ### y = 16

index_seasons = np.zeros(4, dtype = 'object')
index_seasons[0] = np.arange(0, 59, 1)
index_seasons[1] = np.arange(59, 151, 1)
index_seasons[2] = np.arange(151, 243, 1)
index_seasons[3] = np.arange(243, 334, 1)
for i in range(0, y - 1):
	index_seasons[0] = np.concatenate((index_seasons[0], np.arange(334 + 365 * i, 424 + 365 * i, 1)))
	index_seasons[1] = np.concatenate((index_seasons[1], np.arange(424 + 365 * i, 516 + 365 * i, 1)))
	index_seasons[2] = np.concatenate((index_seasons[2], np.arange(516 + 365 * i, 608 + 365 * i, 1)))
	index_seasons[3] = np.concatenate((index_seasons[3], np.arange(608 + 365 * i, 699 + 365 * i, 1)))



P1 = np.zeros((80,80),dtype = 'int')
P2 = np.zeros((80,80),dtype = 'int')
P3 = np.zeros((80,80),dtype = 'int')
P4 = np.zeros((80,80),dtype = 'int')
P5 = np.zeros((80,80),dtype = 'int')



for i in range(3,80):
    for j in range(3, i+1):
        cor = np.zeros(2000, dtype = 'int')
        for k in range(2000):
            ei = np.sort(np.random.choice(index_seasons[2],i,replace=False))
            ej = np.sort(np.random.choice(index_seasons[2],j,replace=False)) 
            cor[k] = calc_ES_ij(ei, ej)
        th05 = st.scoreatpercentile(cor, 95)
        th02 = st.scoreatpercentile(cor, 98)
        th01 = st.scoreatpercentile(cor, 99)
        th005 = st.scoreatpercentile(cor, 99.5)
        th001 = st.scoreatpercentile(cor, 99.9)
        P1[i,j] = th05
        P2[i,j] = th02
        P3[i,j] = th01
        P4[i,j] = th005
        P5[i,j] = th001
    print(i)


# randomly generate EREs for 2000 times in range of 43 years
tm = 10
np.save('rep2000_mnoe80_thresholds_05_tm%d_2_y16'%tm, P1)
np.save('rep2000_mnoe80_thresholds_02_tm%d_2_y16'%tm, P2)
np.save('rep2000_mnoe80_thresholds_01_tm%d_2_y16'%tm, P3)
np.save('rep2000_mnoe80_thresholds_005_tm%d_2_y16'%tm, P4)
np.save('rep2000_mnoe80_thresholds_001_tm%d_2_y16'%tm, P5)