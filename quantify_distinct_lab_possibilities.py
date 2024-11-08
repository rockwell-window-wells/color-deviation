# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 07:58:49 2024

@author: Ryan.Larson

Script for determining how many measurements might return a delta E of less
than a chosen threshold, given a measurement precision (step).

"""

import numpy as np

def get_deltaE(L, a, b, refcolor):
    Lref = refcolor[0]
    aref = refcolor[1]
    bref = refcolor[2]
    deltaE = np.sqrt((L-Lref)**2 + (a-aref)**2 + (b-bref)**2)
    return deltaE


precision = 0.01
deltaEthresh = 0.2

refcolor = (79.96, 1.75, 5.63)

# Determine the color ranges, applying a deltaEthresh in all directions
Lmin = refcolor[0] - deltaEthresh
Lmax = refcolor[0] + deltaEthresh
amin = refcolor[1] - deltaEthresh
amax = refcolor[1] + deltaEthresh
bmin = refcolor[2] - deltaEthresh
bmax = refcolor[2] + deltaEthresh

num_steps = int((Lmax - Lmin) / precision) + 1

Lvals = np.linspace(Lmin, Lmax, num_steps)
avals = np.linspace(amin, amax, num_steps)
bvals = np.linspace(bmin, bmax, num_steps)

count = 0
for L in Lvals:
    for a in avals:
        for b in bvals:
            deltaE = get_deltaE(L, a, b, refcolor)
            
            if deltaE <= deltaEthresh:
                count += 1
                
print(f'{count} values within delta E = {deltaEthresh}')