# UROP2023-ExtremeRainfall
Contains revised code for "Complex networks reveal global pattern of  extreme-rainfall teleconnections"  


all_ere_start.ipynb / extreme rainfall calculator.ipynb  
This file calculates the date of extreme rainfall events at all locations. We first calculate the 95 percentile of all wet days, and then delete consecutive days. 


fig1.ipynb  
Plot figure 1 using cartopy. 


Event_sync.py  
This file aims to calculate the value of event synchronization between each two nodes using the ERE_start_date file. We first use calculate the threshold of each pair of extreme event between two nodes, and see how many pairs satisfy the thresholds. This number is the event synchronization value between these two nodes, and we only save the value if the ES value is larger than the 99.5% null model threshold. In this file, we use both Numba and parallel computation to enhance the calculation speed. 


distance_count.py  
We first use "pos" function to find the corresponding lon and lat of each node, and then use "Hav_distance" function to find the great circle distance between two positions. Then, we define several range for these distances and calculate how many pairs are there in each range. Note that instead of saving counting 1 distance between each two points, we actually save n pairs of such distance between two nodes, where n is the value of ES between these two nodes. We will then use this distance distribution to plot figure 2. 


distance.ipynb  
In this file, we calculates the distance between each two pair of these 576000 nodes. Note that the symmetric property can reduce the computational cost a lot. We save the distances in distance_list. 


distanceKDE.ipynb  
In this file, we first use "distance_list" and KernelDensity to calculate the great circle distance under KDE. We save the result in "logprob.npy". We then use the distance distribution computed using distance_count.py to plot the distritbution of links. We can plot the distribution of links and np.exp(logprob) in the same figure to compare them and find the power law fit for links that have short distances. 


4a.ipynb  
We first find the corresponding indices for SCA and EU, and then compute two time series for these two regions. The n th position in each time series is the number of ERE happen in that region on date n. We then compute the lead-lag correlation between these two time series. 


4b_timeseries.ipynb  
In this file, we include two methods to calculate the time series. The first method is given by author, which is very slow but straightforward. The second method computes t21 and t12 with much less computational cost. Then, these time series are filtered and local maximums that are greater than 95 percentile of all JJA data are chosen. 


4b_plot.ipynb  
We plot the precipitation data of the local maximum dates. 


4d_anomalies_plot.ipynb  
We obtain the data of wind speed. We compute the average wind speed over JJA seasons and subtract if from the average wind speed over local maximum days. This gives the anomalies data, and we plot it to obtain the Rossby-wave-like pattern. 


