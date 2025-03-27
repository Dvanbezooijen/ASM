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
df_raw = load_cpt_data('C:/Users/dylan/Documents/Python/ASM/CPT/CPT000000197792.xml')


#%% COMPUTE BASIC DATA AND PLOTS

#adjust raw dataset (sort by depth and cutoff)
cutoff_end = 5 #last cutoff values should not be read
cutoff_start = 4 #first few values should not be read
df = df_raw.sort_values(by="Depth (m)", ascending=True).iloc[cutoff_start:-cutoff_end].copy()


# Create a figure with 3 subplots (stacked vertically)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)

# Line thickness
line_width = 2  

# Plot 1: Depth vs. Sleeve Friction
axes[0].plot(df["Sleeve Friction (fs, kPa)"], df["Depth (m)"], linewidth=line_width)
axes[0].set_xlabel("Sleeve friction (kPa)")
axes[0].set_ylabel("Depth (m)")
axes[0].grid()
axes[0].invert_yaxis()  # Depth increases downwards
axes[0].set_title("Depth vs. Sleeve Friction")

# Plot 2: Depth vs. Cone Resistance
axes[1].plot(df["Cone Resistance (qc, kPa)"]/1000, df["Depth (m)"], linewidth=line_width)
axes[1].set_xlabel("Cone resistance (MPa)")
axes[1].grid()
axes[1].set_title("Depth vs. Cone Resistance")

# Plot 3: Depth vs. Friction Ratio
axes[2].plot(df["Friction Ratio (%)"], df["Depth (m)"], linewidth=line_width)
axes[2].set_xlabel("Friction Ratio (%)")
axes[2].grid()
axes[2].set_title("Depth vs. Friction Ratio")

# Adjust layout for better readability
plt.tight_layout()
plt.show()

#%% SBT chart

# Load and display the SBT chart as background
img = plt.imread('SBTchart.png')

# Create the scatter plot with switched axes
fig, ax = plt.subplots()

# Create a secondary y-axis using ax.twinx() for displaying the image
ax2 = ax.twinx()

# Plot the image on the secondary axis (background) - Ensure it's behind
ax2.imshow(img, extent=[0, 8, 0.1, 100], aspect='auto', alpha=0.3)

# Plot the scatter plot on the primary axis (foreground, higher zorder)
sc = ax.scatter(df['Friction Ratio (%)']*100, df['Cone Resistance (qc, kPa)']/1000, 
                c=df['Depth (m)'], cmap='viridis', edgecolors='k', s=100, alpha=1.0)

# Set logarithmic scale for the y-axis of the primary axis
ax.set_yscale('log')

# Add labels and title
ax.set_xlabel('Friction Ratio (%)')
ax.set_ylabel('Cone Resistance (qc) [MPa]')

# Set x and y limits for the primary axis (scatter plot)
ax.set_xlim(0, 8)  # x-axis range for friction ratio
ax.set_ylim(0.1, 100)  # Logarithmic scale: set y-limit up to 100 MPa (log scale)

# Color bar based on depth
cbar = plt.colorbar(sc, ax=ax,location='right')
cbar.set_label('Depth (m)')

#hide y-axis
ax2.get_yaxis().set_visible(False)

# Show the plot
plt.show()

#%% COMPUTE SOIL PARAMETERS

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













