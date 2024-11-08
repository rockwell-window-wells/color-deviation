# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:06:20 2024

@author: Ryan.Larson
"""

import numpy as np
import pandas as pd

def get_deltaE(L, a, b, refcolor):
    Lref = refcolor[0]
    aref = refcolor[1]
    bref = refcolor[2]
    deltaE = np.sqrt((L-Lref)**2 + (a-aref)**2 + (b-bref)**2)
    return deltaE

file = "Denali Color Checking.xlsx"

df1 = pd.read_excel(file, skiprows=3)

colnames = ['Date',
            'Truck Number',
            'Well Number',
            'Well Size',
            'UL_L',
            'UL_a',
            'UL_b',
            'LL_L',
            'LL_a',
            'LL_b',
            'UR_L',
            'UR_a',
            'UR_b',
            'LR_L',
            'LR_a',
            'LR_b',
            'Op1_1_L',
            'Op1_1_a',
            'Op1_1_b',
            'Op1_2_L',
            'Op1_2_a',
            'Op1_2_b',
            'Op1_3_L',
            'Op1_3_a',
            'Op1_3_b',
            'Op1_Name',
            'Op2_1_L',
            'Op2_1_a',
            'Op2_1_b',
            'Op2_2_L',
            'Op2_2_a',
            'Op2_2_b',
            'Op2_3_L',
            'Op2_3_a',
            'Op2_3_b',
            'Op2_Name',
            'Raw image file',
            'Thermal shock mask file',
            'Pass (General Inspection)',
            'Thermal shock pixel count',
            '% pixels in thermal shock',
            'Notes'
            ]

df1.columns = colnames

#%% Produce a new dataframe based on delta E values relative to a reference
refLAB = (79.96, 1.75, 5.63)
color_groups = [['UL_L', 'UL_a', 'UL_b'],
                ['LL_L', 'LL_a', 'LL_b',],
                ['UR_L', 'UR_a', 'UR_b',],
                ['LR_L', 'LR_a', 'LR_b',],
                ['Op1_1_L', 'Op1_1_a', 'Op1_1_b',],
                ['Op1_2_L', 'Op1_2_a', 'Op1_2_b',],
                ['Op1_3_L', 'Op1_3_a', 'Op1_3_b',],
                ['Op2_1_L', 'Op2_1_a', 'Op2_1_b',],
                ['Op2_2_L', 'Op2_2_a', 'Op2_2_b',],
                ['Op2_3_L', 'Op2_3_a', 'Op2_3_b',],
                ]

keep_cols1 = ['Date',
            'Truck Number',
            'Well Number',
            'Well Size']
keep_cols2 = ['Raw image file',
            'Thermal shock mask file',
            'Pass (General Inspection)',
            'Thermal shock pixel count',
            '% pixels in thermal shock',
            'Notes']

df2 = df1[keep_cols1].copy()

# Iterate through the color groups and get the prefixes
prefixes = []
for group in color_groups:
    prefix = group[0][:-2]
    prefixes.append(prefix)

# Iterate through the rows of the color groups and calculate the delta E
for i, group in enumerate(color_groups):
    prefix = prefixes[i]
    
    deltaE_list = []
    df_slice = df1[group]
    for j, row in df_slice.iterrows():
        L = row.iloc[0]
        a = row.iloc[1]
        b = row.iloc[2]
        
        deltaE = get_deltaE(L, a, b, refLAB)
        deltaE_list.append(deltaE)
        
    df2[prefix] = deltaE_list


df2[keep_cols2] = df1[keep_cols2].copy()

###############################################################################
# One good measure would be the deltaE between the base color and the thermal shock
###############################################################################

#%% Produce a new dataframe that gets the average and max delta E values for
# each category, per well
df3 = df1[keep_cols1].copy()
measurement_groups = [['UL', 'LL', 'UR', 'LR'],
                      ['Op1_1', 'Op1_2', 'Op1_3', 'Op2_1', 'Op2_2', 'Op2_3']]

for i, group in enumerate(measurement_groups):
    avg_deltaE = []
    max_deltaE = []
    df_slice = df2[group]
    for j, row in df_slice.iterrows():
        avg = row.mean()
        maxval = row.max()
        avg_deltaE.append(avg)
        max_deltaE.append(maxval)
        
    if i == 0:
        df3['Base Color Avg Delta E'] = avg_deltaE
        df3['Base Color Max Delta E'] = max_deltaE
    else:
        df3['Thermal Shock Avg Delta E'] = avg_deltaE
        df3['Thermal Shock Max Delta E'] = max_deltaE
        
df3[keep_cols2] = df2[keep_cols2].copy()
        