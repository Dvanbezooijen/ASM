import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from Enschede_read_data import enschede_read_data

# List of names for the DataFrames
list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "7_630N","8_810N", "9D_180N", "10D_720N", "11D_1260N"]
all_df = enschede_read_data(list_of_names)

size_shear_stress = 0.0036 #m^2

# Heights for each test (in mm)
heights = {
    "1_180N": 28, "2_450N": 25, "4_990N": 24, "5_1260N": 29,
    "6_720N": 25.5, "7_630N": 27, "8_810N": 25.5, "9D_180N": 24, "10D_720N": 24, "11D_1260N": 21
}

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
"""
# Function to find stabilization point using Savitzky-Golay filtering
def find_stabilization_point(df, window=11, polyorder=3, threshold=0.1):
    df = df.iloc[int(0.1 * len(df)):]  # Exclude the first 10% of data
    df["Smoothed Stress"] = savgol_filter(df["Stress (kPa)"], window, polyorder)
    df["First Derivative"] = np.gradient(df["Smoothed Stress"])
    df["Second Derivative"] = np.gradient(df["First Derivative"])
    
    for i in range(len(df)):
        if abs(df["First Derivative"].iloc[i]) < threshold and df["Second Derivative"].iloc[i] > -threshold:
            return df.iloc[i]["Stress (kPa)"], df.iloc[i]["Shear Strain"]
    
    return df.iloc[-1]["Stress (kPa)"], df.iloc[-1]["Shear Strain"]
"""
# Create lists to store stress values and applied stress values
stress_values = []
applied_stress_values = []

# Define the custom indices for selecting specific points in each dataset
custom_indices = [0.075 , 0.125, 0.125, 0.085, 0.080, 0.080, 0.085]  # Example list of indices

# Iterate through the first 6 DataFrames and custom indices
for idx, name in enumerate(list_of_names[:7]):
    df = all_df[name]
    df["Shear Strain"] = df["Horizontal Displacement (mm)"] / heights[name]
    
    # Calculate applied stress in kPa
    applied_stress = (int(name.split('_')[1].replace('N', '')) / size_shear_stress) / 100  # kPa
    applied_stress_values.append(applied_stress)
    
    # Use the custom index for selecting the stress value
    custom_index = custom_indices[idx]
    
    # Calculate the exact row index based on the custom index (no rounding)
    row_index = int(custom_index * len(df))  # Multiply custom index by the length of the dataframe
    
    # Ensure the index is within the valid range
    row_index = min(row_index, len(df) - 1)
    
    # Select the stress value at the calculated index
    stress = df.iloc[row_index]["Stress (kPa)"]
    
    # Append the stress value to the list
    stress_values.append(stress)
    
# Create a DataFrame for seaborn
data = pd.DataFrame({
    'Applied Stress (kPa)': applied_stress_values,
    'Shear Stress (kPa)': stress_values
})

# Plot the regression for the first 6 names
sns.regplot(x='Applied Stress (kPa)', y='Shear Stress (kPa)', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[0])

# Get the slope and calculate friction angle
slope, intercept = np.polyfit(data['Applied Stress (kPa)'], data['Shear Stress (kPa)'], 1)
phi_cs = np.arctan(slope) * (180/np.pi)
axs[0].text(0.1, 0.9, f'Friction angle = {phi_cs:.4f}°', transform=axs[0].transAxes, fontsize=12, color='black', ha='left')

# Labels and title
axs[0].set_xlabel('Applied Stress (kPa)')
axs[0].set_ylabel('Shear Stress (kPa)')
axs[0].set_title('Shear Stress vs Applied Stress (normal density) with Regression Line')
axs[0].grid(True)

# Repeat for the last 3 names
stress_values = []
applied_stress_values = []

for name in list_of_names[7:]:
    df = all_df[name]
    df["Shear Strain"] = df["Horizontal Displacement (mm)"] / heights[name]
    #stress, _ = 0#find_stabilization_point(df)
    stress = 1
    applied_stress = (int(name.split('_')[1].replace('N', '')) / size_shear_stress) / 100  # kPa
    stress_values.append(stress)
    applied_stress_values.append(applied_stress)

# Create a DataFrame
data = pd.DataFrame({
    'Applied Stress (kPa)': applied_stress_values,
    'Shear Stress (kPa)': stress_values
})

# Plot regression for last 3 names
sns.regplot(x='Applied Stress (kPa)', y='Shear Stress (kPa)', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[1])

# Get slope and calculate friction angle
slope, intercept = np.polyfit(data['Applied Stress (kPa)'], data['Shear Stress (kPa)'], 1)
phi_cs = np.arctan(slope) * (180/np.pi)
axs[1].text(0.1, 0.9, f'Friction angle = {phi_cs:.4f}°', transform=axs[1].transAxes, fontsize=12, color='black', ha='left')

# Labels and title
axs[1].set_xlabel('Applied Stress (kPa)')
axs[1].set_ylabel('Shear Stress (kPa)')
axs[1].set_title('Shear Stress vs Applied Stress (different density) with Regression Line')
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Create second plot for Stress-Strain graph
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First subplot
for name in list_of_names[:7]:
    df = all_df[name]
    axs[0].plot(df["Shear Strain"], df["Stress (kPa)"], label=f'{name} Data')

axs[0].set_xlabel('Shear Strain')
axs[0].set_ylabel('Stress (kPa)')
axs[0].set_title('Stress vs Shear Strain (normal density)')
axs[0].legend()
axs[0].grid(True)

# Second subplot
for name in list_of_names[7:]:
    df = all_df[name]
    axs[1].plot(df["Shear Strain"], df["Stress (kPa)"], label=f'{name} Data')

axs[1].set_xlabel('Shear Strain')
axs[1].set_ylabel('Stress (kPa)')
axs[1].set_title('Stress vs Shear Strain (different density)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First subplot: Stress-Strain graph for the first 7 names
for name in list_of_names[:7]:
    df = all_df[name]
    axs[0].plot(df["Shear Strain"], df["Stress (kPa)"], label=f'{name} Data')
    # Calculate the stabilized point for the first 7 names
    stabilized_stress, stabilized_strain = find_stabilization_point(df)
    # Plot the stabilized point on the graph
    axs[0].plot(stabilized_strain, stabilized_stress, 'ro')  # Red point for stabilization

axs[0].set_xlabel('Shear Strain')
axs[0].set_ylabel('Stress (kPa)')
axs[0].set_title('Stress vs Shear Strain (normal density)')
axs[0].legend()
axs[0].grid(True)

# Second subplot: Stress-Strain graph for the last 3 names
for name in list_of_names[7:]:
    df = all_df[name]
    axs[1].plot(df["Shear Strain"], df["Stress (kPa)"], label=f'{name} Data')
    # Calculate the stabilized point for the last 3 names
    stabilized_stress, stabilized_strain = find_stabilization_point(df)
    # Plot the stabilized point on the graph
    axs[1].plot(stabilized_strain, stabilized_stress, 'ro')  # Red point for stabilization

# Labels and title for second plot
axs[1].set_xlabel('Shear Strain')
axs[1].set_ylabel('Stress (kPa)')
axs[1].set_title('Stress vs Shear Strain (different density)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
