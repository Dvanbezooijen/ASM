import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from Enschede_read_data import enschede_read_data

# List of names for the DataFrames
list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "8_810N", "9D_180N", "10D_720N", "11D_1260N"]
all_df = enschede_read_data(list_of_names)

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Function to find stabilization point using Savitzky-Golay filtering
def find_stabilization_point(df, window=11, polyorder=3, threshold=0.1):
    df = df.iloc[int(0.1 * len(df)):]  # Exclude the first 10% of data
    df["Smoothed Stress"] = savgol_filter(df["Stress (kPa)"], window, polyorder)
    df["First Derivative"] = np.gradient(df["Smoothed Stress"])
    df["Second Derivative"] = np.gradient(df["First Derivative"])
    
    for i in range(len(df)):
        if abs(df["First Derivative"].iloc[i]) < threshold and df["Second Derivative"].iloc[i] > -threshold:
            return df.iloc[i]["Stress (kPa)"], df.iloc[i]["Horizontal Displacement (mm)"]
    
    return df.iloc[-1]["Stress (kPa)"], df.iloc[-1]["Horizontal Displacement (mm)"]

# Create lists to store stress values and corresponding applied stress (N values) for the first 6 names
stress_values = []
applied_stress_values = []

# Iterate through the first 6 DataFrames
for name in list_of_names[:6]:
    df = all_df[name]
    stress, _ = find_stabilization_point(df)
    applied_stress = int(name.split('_')[1].replace('N', ''))  
    stress_values.append(stress)
    applied_stress_values.append(applied_stress)

# Create a DataFrame for seaborn
data = pd.DataFrame({
    'Applied Stress (N)': applied_stress_values,
    'Shear Stress (kPa)': stress_values
})

# Plot the regression for the first 6 names
sns.regplot(x='Applied Stress (N)', y='Shear Stress (kPa)', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[0])

# Get the slope and calculate friction angle
slope, intercept = np.polyfit(data['Applied Stress (N)'], data['Shear Stress (kPa)'], 1)
phi_cs = np.arcsin(3 * slope/(6 + slope)) * (180/np.pi)
axs[0].text(0.1, 0.9, f'Friction angle = {phi_cs:.4f}°', transform=axs[0].transAxes, fontsize=12, color='black', ha='left')

# Labels and title
axs[0].set_xlabel('Applied Stress (N)')
axs[0].set_ylabel('Shear Stress (kPa)')
axs[0].set_title('Shear Stress vs Applied Stress (normal density) with Regression Line')
axs[0].grid(True)

# Repeat for the last 3 names
stress_values = []
applied_stress_values = []

for name in list_of_names[6:]:
    df = all_df[name]
    stress, _ = find_stabilization_point(df)
    applied_stress = int(name.split('_')[1].replace('N', ''))  
    stress_values.append(stress)
    applied_stress_values.append(applied_stress)

# Create a DataFrame
data = pd.DataFrame({
    'Applied Stress (N)': applied_stress_values,
    'Shear Stress (kPa)': stress_values
})

# Plot regression for last 3 names
sns.regplot(x='Applied Stress (N)', y='Shear Stress (kPa)', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[1])

# Get slope and calculate friction angle
slope, intercept = np.polyfit(data['Applied Stress (N)'], data['Shear Stress (kPa)'], 1)
phi_cs = np.arcsin(3 * slope/(6 + slope)) * (180/np.pi)
axs[1].text(0.1, 0.9, f'Friction angle = {phi_cs:.4f}°', transform=axs[1].transAxes, fontsize=12, color='black', ha='left')

# Labels and title
axs[1].set_xlabel('Applied Stress (N)')
axs[1].set_ylabel('Shear Stress (kPa)')
axs[1].set_title('Shear Stress vs Applied Stress (different density) with Regression Line')
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Create second plot
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First subplot (first 6 data sets)
for i, name in enumerate(list_of_names[:6]):
    df = all_df[name]

    # Initial stress and displacement
    initial_stress = min(df["Stress (kPa)"])
    initial_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == initial_stress].iloc[0]

    # Maximum stress and displacement
    maximum_stress = max(df["Stress (kPa)"])
    maximum_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == maximum_stress].iloc[0]

    # Stabilization point
    stab_stress, stab_displacement = find_stabilization_point(df)

    # Plot the data on the first subplot
    axs[0].plot(df["Horizontal Displacement (mm)"], df["Stress (kPa)"], label=f'{name} Data')

    # Plot red line from initial to stabilization point
    axs[0].plot([initial_displacement, stab_displacement], [initial_stress, stab_stress], 'r-', label=f'{name} Initial to Stab')

    # Plot red dot at the stabilization point
    axs[0].plot(stab_displacement, stab_stress, 'ro', label=f'{name} Stab Point')

# First subplot labels and title
axs[0].set_xlabel('Horizontal Displacement (mm)')
axs[0].set_ylabel('Stress (kPa)')
axs[0].set_title('Stress vs Horizontal Displacement (normal density)')
axs[0].legend(loc='best')
axs[0].grid(True)

# Second subplot (remaining data sets)
for i, name in enumerate(list_of_names[6:]):
    df = all_df[name]

    # Initial stress and displacement
    initial_stress = min(df["Stress (kPa)"])
    initial_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == initial_stress].iloc[0]

    # Maximum stress and displacement
    maximum_stress = max(df["Stress (kPa)"])
    maximum_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == maximum_stress].iloc[0]

    # Stabilization point
    stab_stress, stab_displacement = find_stabilization_point(df)

    # Plot the data on the second subplot
    axs[1].plot(df["Horizontal Displacement (mm)"], df["Stress (kPa)"], label=f'{name} Data')

    # Plot red line from initial to stabilization point
    axs[1].plot([initial_displacement, stab_displacement], [initial_stress, stab_stress], 'r-', label=f'{name} Initial to Stab')

    # Plot red dot at the stabilization point
    axs[1].plot(stab_displacement, stab_stress, 'ro', label=f'{name} Stab Point')

# Second subplot labels and title
axs[1].set_xlabel('Horizontal Displacement (mm)')
axs[1].set_ylabel('Stress (kPa)')
axs[1].set_title('Stress vs Horizontal Displacement (different density)')
axs[1].legend(loc='best')
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()