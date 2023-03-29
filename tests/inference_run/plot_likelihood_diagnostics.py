#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:20:05 2023

@author: bas and chatGPT :)
"""

import dill as pickle
import matplotlib.pyplot as plt

numenergies=32

with open(f'run_A5/LikelihoodDiagnostics_numenergies={numenergies}.pkl', 'rb') as file:
     ldict = pickle.load(file)
     ldict.popitem()

# reduce time
diff =  ldict[1]['starttime']
for i in ldict:
    ldict[i]['starttime'] =  ldict[i]['starttime'] - diff
    ldict[i]['endtime']   =  ldict[i]['endtime'] - diff


# Extract start and end times
start_times = [ldict[i]['starttime'] for i in ldict]
end_times = [ldict[i]['endtime'] for i in ldict]

# Create a list of time points where the likelihood is being computed
times = []
for i in range(len(start_times)):
    times.append(start_times[i])
    times.append(end_times[i])

times.sort()  # sort the times in ascending order

# Create a list of values for the y-axis
y = []
for i in range(len(times)):
    if i % 2 == 0:
        y.append(1)  # likelihood is being computed
    else:
        y.append(0)  # likelihood is not being computed


# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.step(times, y, where='post')
ax.set_xlabel('Time')
ax.set_ylabel('Likelihood')
ax.set_ylim(-0.1, 1.1)  # set y-axis limits to be slightly larger than 0 and 1

# Label the plot with the appropriate key for each likelihood evaluation
for i, t in enumerate(times):
    if i % 2 == 0:
        key = [k for k, v in ldict.items() if v['starttime'] == t][0]
        ax.annotate(str(key), xy=(t, y[i]), xytext=(t, 1.1), ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='black', lw=0.5))
        
plt.show()

# print time between likelihood evals
# prev_endtime = None
# for key in sorted(ldict.keys()):
#     if prev_endtime is not None:
#         delta_time = ldict[key]['starttime'] - prev_endtime
#         print(f'Time between likelihood {key-1} and {key}: {delta_time:.3f}')
#     prev_endtime = ldict[key]['endtime']
    
keys = sorted(ldict.keys())
max_gap = 0
n = None
for i in range(len(keys)-1):
    if 'endtime' in ldict[keys[i]] and 'starttime' in ldict[keys[i+1]]:
        gap = ldict[keys[i+1]]['starttime'] - ldict[keys[i]]['endtime']
        if gap > max_gap:
            max_gap = gap
            n = i+1
if n is None:
    n = 0
total_time = ldict[keys[-1]]['endtime'] - ldict[keys[n]]['starttime']
likelihood_time = 0
for i in range(len(keys)-1):
    if i >= n:
        if 'deltatime' in ldict[keys[i+1]]:
            if 'endtime' in ldict[keys[i]] and 'starttime' in ldict[keys[i+1]]:
                likelihood_time += ldict[keys[i+1]]['deltatime']
            elif 'starttime' in ldict[keys[i+1]]:
                prev_endtime = ldict[keys[i+1]]['starttime']
                for j in range(i, -1, -1):
                    if 'endtime' in ldict[keys[j]]:
                        prev_endtime = ldict[keys[j]]['endtime']
                        break
                gap = ldict[keys[i+1]]['starttime'] - prev_endtime
                if gap > 0:
                    likelihood_time += ldict[keys[i+1]]['deltatime']
percentage_likelihood = 100 * likelihood_time / total_time
print(f"The nth iteration is {n}")
print(f"Percentage of time spent doing likelihood computations after the {n-1}th iteration: {percentage_likelihood:.2f}%")