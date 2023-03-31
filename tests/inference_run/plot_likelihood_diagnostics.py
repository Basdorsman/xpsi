#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:20:05 2023

@author: bas and chatGPT :)
"""

import dill as pickle
import matplotlib.pyplot as plt

numenergies=32
atmosphere_type = 'A'
n_params = 4
xpsi_size = 4

ldict = {}

folderstring = f'run_{atmosphere_type}{n_params}'

fig, axes = plt.subplots(nrows=xpsi_size, ncols=1, figsize=(20, 2*xpsi_size), sharex=True)
for ax, rank in zip(axes, range(xpsi_size)):
    with open(f'{folderstring}/LikelihoodDiagnostics_ne={numenergies}_rank={rank}.pkl', 'rb') as file:
         (ldict[rank], runtime_params) = pickle.load(file)

    tmpdict = ldict[rank]

    # reduce time
    diff =  tmpdict[0]['starttime']
    start_times = []
    end_times=[]
    times = []
    for call in tmpdict:
        tmpdict[call]['starttime'] =  tmpdict[call]['starttime'] - diff
        tmpdict[call]['endtime']   =  tmpdict[call]['endtime'] - diff


        # Extract start and end times
        start_times.append(tmpdict[call]['starttime']) 
        end_times.append(tmpdict[call]['endtime'])

        # Create a list of time points where the likelihood is being computed
        times.append(start_times[call])
        times.append(end_times[call])

    times.sort()  # sort the times in ascending order

    # Create a list of values for the y-axis
    y = []
    for i in range(len(times)):
        if i % 2 == 0:
            y.append(1)  # likelihood is being computed
        else:
            y.append(0)  # likelihood is not being computed


    # Create the plot
    ax.step(times, y, where='post')
    ax.set_xlabel('Time')
    ax.set_ylabel('Likelihood')
    ax.set_ylim(-0.1, 1.1)  # set y-axis limits to be slightly larger than 0 and 1

    # Label the plot with the appropriate key for each likelihood evaluation
    for i, t in enumerate(times):
        if i % 2 == 0:
            key = [k for k, v in tmpdict.items() if v['starttime'] == t][0]
            ax.annotate(str(key), xy=(t, y[i]), xytext=(t, 1.1), ha='center', va='top', fontsize=5,
                        bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='black', lw=0.5))

    # Title with percentage time calculating likelihoods
    # Start after gap at the start (not part of sampling)
    keys = sorted(tmpdict.keys())
    max_gap = 0
    n = None
    for i in range(len(keys)-1):
        if 'endtime' in tmpdict[keys[i]] and 'starttime' in tmpdict[keys[i+1]]:
            gap = tmpdict[keys[i+1]]['starttime'] - tmpdict[keys[i]]['endtime']
            if gap > max_gap:
                max_gap = gap
                n = i+1
    if n is None:
        n = 0
    total_time = tmpdict[keys[-1]]['endtime'] - tmpdict[keys[n]]['starttime']
    likelihood_time = 0
    for i in range(len(keys)-1):
        if i >= n:
            if 'deltatime' in tmpdict[keys[i+1]]:
                if 'endtime' in tmpdict[keys[i]] and 'starttime' in tmpdict[keys[i+1]]:
                    likelihood_time += tmpdict[keys[i+1]]['deltatime']
                elif 'starttime' in tmpdict[keys[i+1]]:
                    prev_endtime = tmpdict[keys[i+1]]['starttime']
                    for j in range(i, -1, -1):
                        if 'endtime' in tmpdict[keys[j]]:
                            prev_endtime = tmpdict[keys[j]]['endtime']
                            break
                    gap = tmpdict[keys[i+1]]['starttime'] - prev_endtime
                    if gap > 0:
                        likelihood_time += tmpdict[keys[i+1]]['deltatime']
    percentage_likelihood = 100 * likelihood_time / total_time
    
    ax.set_title(f"Percentage of time spent doing likelihood computations starting at n={n}: {percentage_likelihood:.2f}%")

fig.tight_layout()
fig.savefig(f'{folderstring}/diagnostics_{atmosphere_type}{n_params}_size={xpsi_size}.png',dpi=300)