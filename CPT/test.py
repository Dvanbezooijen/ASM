# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:38:01 2025
@author: Dylan van Bezooijen
Function: Analyse data from CPT 
"""
from loader import load_cpt_data
import matplotlib.pyplot as plt
import numpy as np

#use loader to load relevant data into pandas
df_raw = load_cpt_data('CPT/CPT000000197792.xml')

#adjust raw dataset (sort by depth and cutoff)
cutoff_end = 5 #last cutoff values should not be read
cutoff_start = 4 #first few values should not be read
df = df_raw.sort_values(by="Depth (m)", ascending=True).iloc[cutoff_start:-cutoff_end].copy()

#unit weight of soil
unit_weight_soil  = 18 # kN/m3
unit_weight_w = 10 #kN/m3

#calculate effective vertical stress
df['Effective_vertical_stress (kPa)'] = df['Depth (m)'] * (unit_weight_soil - unit_weight_w)

#friction angle
df['Friction angle'] = np.arctan(0.1 + 0.38 * np.log(df['Cone Resistance (qc, kPa)']/df['Effective_vertical_stress (kPa)']))
#Equivelant modulus of elasticity
df['Equivalant modulus of Elasticity'] = 7 * df['Cone Resistance (qc, kPa)'] #clay

#atmospheric pressure
Pa = 100 #kPa

#relative density (Lancelotta)
df['Relative Density (%)'] = 68*(np.log(df['Cone Resistance (qc, kPa)']/
                                        np.sqrt(Pa*df['Effective_vertical_stress (kPa)']))-1)
#bearing capacity factor
Nk = 18.3

#undrained shear strength
df['Undrained shear strength (kPa)'] = (df['Cone Resistance (qc, kPa)'] - df['Effective_vertical_stress (kPa)']/
                                        Nk)

# Given data
depth = np.array([2.5, 10, 20])  # Depth in meters
cone_resistance = np.array([0.2, 0.6, 1.15]) * 1000  # Convert MPa to KPa
friction_ratio = np.array([0.04, 0.11, 0.20]) * 100  # Convert to %

# Reference values
gamma_sat_ref = 15  # kN/m³
qc_ref = 1000  # KPa
Rf_ref = 6.0  # %
beta = 1.25  # Inclination factor

# Constants
gamma_w = 9.81  # kN/m³ (Unit weight of water)
atmospheric_pressure = 1000 #KPa 

# Compute gamma_sat
gamma_sat = gamma_sat_ref - beta * (np.log10(qc_ref / cone_resistance) / np.log10(Rf_ref / friction_ratio))

# Compute total vertical stress (sigma_v)
sigma_v = np.cumsum(gamma_sat * np.diff(np.hstack(([0], depth))))

# Compute pore pressure (u)
u = gamma_w * depth

# Compute effective vertical stress (sigma'_v)
sigma_v_prime = sigma_v - u

# Compute friction angle           
friction_angle = np.degrees(np.arctan(0.1 + 0.38 * np.log10(cone_resistance / sigma_v_prime)))  # (inc bron)
dilation_angle = np.maximum(friction_angle - 30, 0)    ###acccording to bron

# Compute modulus of elasticity
E_s = 7 * cone_resistance                               #according to bron 

# Compute undrained shear strength
s_u = (cone_resistance - sigma_v) / 18.3

# Compute relativy density
q_c1 = cone_resistance*np.sqrt(atmospheric_pressure/sigma_v_prime)
relative_density = np.sqrt((1/305)*(q_c1/atmospheric_pressure))

# Compute the Ellipse Aspect Ratio from Mohr-Coulomb theory
phi_rad = np.radians(friction_angle)  # Convert friction angle to radians
R = (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))
theoretical_aspect_ratio = (1 + R) / (R - 1)

# Soil Sensitivity
soil_sensitivity  = 7.1 * friction_ratio

#volumetric strain 
epselon = s_u/cone_resistance

print("\nDepth (m) | Cone Resistance (kPa) | Friction Ratio (%) | Total Stress (kPa) | Pore Pressure (kPa) | Effective Stress (kPa) | Friction Angle (°) | Dilation Angle (°) | Modulus of Elasticity (kPa) | Undrained Shear Strength (kPa) | Relative Density | Aspect Ratio | Sensitivity | Volumetric Strain ")
print("-" * 220)

for d, cr, fr, sv, u_, svp, phi, dil, Es, su, rd, ar, sens, eps in zip(
    depth, cone_resistance, friction_ratio, sigma_v, u, sigma_v_prime,
    friction_angle, dilation_angle, E_s, s_u, relative_density,
    theoretical_aspect_ratio, soil_sensitivity, epselon
):
    print(f"{d:8.2f} | {cr:18.2f} | {fr:18.2f} | {sv:18.2f} | {u_:18.2f} | {svp:22.2f} | {phi:15.2f} | {dil:18.2f} | {Es:27.2f} | {su:29.2f} | {rd:16.4f} | {ar:12.2f} | {sens:11.2f} | {eps:17.6f}")
