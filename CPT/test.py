# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:38:01 2025
@author: Dylan van Bezooijen
Function: Analyse data from CPT 
"""
#%% IMPORT PACKAGES
from loader import load_cpt_data
import matplotlib.pyplot as plt
import numpy as np

#%% LOAD DATA & COMPUTE BASIC VARIABLES

#use loader to load relevant data into pandas
df_raw = load_cpt_data('CPT/CPT000000197792.xml')

#%% COMPUTE BASIC DATA AND PLOTS

#adjust raw dataset (sort by depth and cutoff)
cutoff_end = 5 #last cutoff values should not be read
cutoff_start = 4 #first few values should not be read
df = df_raw.sort_values(by="Depth (m)", ascending=True).iloc[cutoff_start:-cutoff_end].copy()

# Compute effective vertical stress
unit_weight_soil  = 18  # kN/m3
unit_weight_w = 10  # kN/m3
df['Effective_vertical_stress (kPa)'] = df['Depth (m)'] * (unit_weight_soil - unit_weight_w)

# Compute Friction Angle (φ')
df["Friction angle (φ')"] = np.degrees(np.arctan(0.1 + 0.38 * np.log(df['Cone Resistance (qc, kPa)'] / df['Effective_vertical_stress (kPa)'])))

# Compute Dilation Angle (ψ)
df['Dilation Angle (ψ)'] = df["Friction angle (φ')"] - 30

# Compute Isotropic Hardening Modulus (H) - using empirical factor
df['Isotropic Hardening Modulus (kPa)'] = 3.0 * df['Cone Resistance (qc, kPa)']

# Compute Maximum Plastic Volumetric Strain (assumed relation to Friction Ratio)
df['Maximum Plastic Volumetric Strain'] = 0.01 * df['Friction Ratio (%)']

# Compute Ellipse Aspect Ratio (approximated as function of depth)
df['Ellipse Aspect Ratio'] = 1.5 - 0.05 * df['Depth (m)']

# Compute Young's Modulus (E) - empirical relation
df["Young's Modulus (kPa)"] = 7 * df['Cone Resistance (qc, kPa)']

# Compute Average Density (kg/m³) - approximated based on soil type and cone resistance
df['Average Density (kg/m³)'] = 1800 + 0.1 * df['Cone Resistance (qc, kPa)']

# Display computed parameters
print(df[['Depth (m)', "Friction angle (φ')", 'Dilation Angle (ψ)', 'Isotropic Hardening Modulus (kPa)',
          'Maximum Plastic Volumetric Strain', 'Ellipse Aspect Ratio', "Young's Modulus (kPa)", 'Average Density (kg/m³)']])
