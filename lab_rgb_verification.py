# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:08:32 2024

@author: Ryan.Larson
"""

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000

APcolor1 = LabColor(lab_l=79.33, lab_a=1.18, lab_b=5.14, observer='10', illuminant='d65')
APcolor2 = LabColor(lab_l=79.69, lab_a=1.75, lab_b=5.75, observer='10', illuminant='d65')

RWcolor1 = LabColor(lab_l=49.09, lab_a=14.23, lab_b=12.66, illuminant='d65')
RWcolor2 = LabColor(lab_l=49.09+0.51, lab_a=14.23-0.36, lab_b=12.66+1.27, illuminant='d65')

# Calculate delta E using different formulas
delta_e_1976 = delta_e_cie1976(RWcolor1, RWcolor2)
delta_e_1994 = delta_e_cie1994(RWcolor1, RWcolor2)
delta_e_2000 = delta_e_cie2000(RWcolor1, RWcolor2)
# delta_e_1976 = delta_e_cie1976(APcolor1, APcolor2)
# delta_e_1994 = delta_e_cie1994(APcolor1, APcolor2)
# delta_e_2000 = delta_e_cie2000(APcolor1, APcolor2)

# Print results
print(f"Delta E (CIE1976): {delta_e_1976}")
print(f"Delta E (CIE1994): {delta_e_1994}")
print(f"Delta E (CIE2000): {delta_e_2000}")


# AP and RW are both using delta_e_1976 standard to calculate delta E values,
# which may contribute to inaccuracies around the neutral color space.