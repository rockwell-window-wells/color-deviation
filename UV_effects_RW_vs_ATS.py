# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 08:41:28 2025

@author: Ryan.Larson
"""

import numpy as np
import pandas as pd

file = "RW_vs_ATS.xlsx"

df = pd.read_excel(file)

df_RW = df[df["Location"] == "Rockwell"]
df_ATS = df[df["Location"] == "ATS"]

for i, row in df_RW.iterrows():
    L_RW = row.loc['L Baseline']
    A_RW = row.loc['A Baseline']
    B_RW = row.loc['B Baseline']

    # print(f'({L_RW} {A_RW} {B_RW})')
