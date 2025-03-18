import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Enschede_read_data import enschede_read_data

# List of names for the DataFrames
list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "8_810N", "9D_180N", "10D_720N", "11D_1260N"]
all_df = enschede_read_data(list_of_names)


# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Create a list to store maximum stress values and corresponding applied stress (N values) for the first 6 names
max_stress_values = []
applied_stress_values = []

# Iterate through the first 6 DataFrames and extract the required data
for name in list_of_names[:6]:
    df = all_df[name]
    
    # Extract maximum stress and corresponding displacement
    maximum_stress = max(df["Stress (kPa)"])
    
    # Extract the applied stress from the DataFrame name (e.g., "1_180N" -> 180N)
    applied_stress = int(name.split('_')[1].replace('N', ''))  # Extract the number and convert to integer
    
    # Append the values to the lists
    max_stress_values.append(maximum_stress)
    applied_stress_values.append(applied_stress)

# Create a DataFrame for seaborn
data = pd.DataFrame({
    'Applied Stress (N)': applied_stress_values,
    'Maximum Shear Stress (kPa)': max_stress_values
})

# Plot the regression for the first 6 names on the first subplot
sns.regplot(x='Applied Stress (N)', y='Maximum Shear Stress (kPa)', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[0])

# Get the slope (M) and intercept for the first plot
slope, intercept = np.polyfit(data['Applied Stress (N)'], data['Maximum Shear Stress (kPa)'], 1)
phi_cs = np.arcsin(3 * slope/(6 + slope)) * ( 180/np.pi)
# Add the slope (M) to the second plot
slope_text = f'Friction angle = {phi_cs:.4f}°'
axs[0].text(0.1, 0.9, slope_text, transform=axs[0].transAxes, fontsize=12, color='black', ha='left')

# Add labels and title for the first subplot
axs[0].set_xlabel('Applied Stress (N)')
axs[0].set_ylabel('Maximum Shear Stress (kPa)')
axs[0].set_title('Maximum Shear Stress vs Applied Stress (normal density) with Regression Line')
axs[0].grid(True)

# Create a list to store maximum stress values and corresponding applied stress (N values) for the last 3 names
max_stress_values = []
applied_stress_values = []

# Iterate through the last 3 DataFrames and extract the required data
for name in list_of_names[6:]:
    df = all_df[name]
    
    # Extract maximum stress and corresponding displacement
    maximum_stress = max(df["Stress (kPa)"])
    
    # Extract the applied stress from the DataFrame name (e.g., "9D_180N" -> 180N)
    applied_stress = int(name.split('_')[1].replace('N', ''))  # Extract the number and convert to integer
    
    # Append the values to the lists
    max_stress_values.append(maximum_stress)
    applied_stress_values.append(applied_stress)

# Create a DataFrame for seaborn
data = pd.DataFrame({
    'Applied Stress (N)': applied_stress_values,
    'Maximum Shear Stress (kPa)': max_stress_values
})

# Plot the regression for the last 3 names on the second subplot
sns.regplot(x='Applied Stress (N)', y='Maximum Shear Stress (kPa)', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[1])

# Get the slope (M) and intercept for the second plot
slope, intercept = np.polyfit(data['Applied Stress (N)'], data['Maximum Shear Stress (kPa)'], 1)
phi_cs = np.arcsin(3 * slope/(6 + slope)) * ( 180/np.pi)
# Add the slope (M) to the second plot
slope_text = f'Friction angle = {phi_cs:.4f}°'
axs[1].text(0.1, 0.9, slope_text, transform=axs[1].transAxes, fontsize=12, color='black', ha='left')

# Add labels and title for the second subplot
axs[1].set_xlabel('Applied Stress (N)')
axs[1].set_ylabel('Maximum Shear Stress (kPa)')
axs[1].set_title('Maximum Shear Stress vs Applied Stress (different density) with Regression Line')
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First subplot (first 8 data sets)
for i, name in enumerate(list_of_names[:6]):
    df = all_df[name]
    
    # Initial and maximum stress values and corresponding displacements
    initial_stress = min(df["Stress (kPa)"])
    initial_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == initial_stress].iloc[0]

    maximum_stress = max(df["Stress (kPa)"])
    maximum_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == maximum_stress].iloc[0]

    # Calculate the slope of the line
    slope = (maximum_displacement - initial_displacement) / (maximum_stress - initial_stress)

    # Plot the data on the first subplot
    axs[0].plot(df["Horizontal Displacement (mm)"], df["Stress (kPa)"], label=f'{name} Data')

    # Plot the line from initial stress to maximum stress
    axs[0].plot([initial_displacement, maximum_displacement], [initial_stress, maximum_stress], 'ro-', label=f'{name} Stress Displacement Line')

# First subplot labels and title
axs[0].set_xlabel('Horizontal Displacement (mm)')
axs[0].set_ylabel('Stress (kPa)')
axs[0].set_title('Stress vs Horizontal Displacement (normal density)')
axs[0].legend(loc='best')
axs[0].grid(True)

# Second subplot (last 3 data sets)
for i, name in enumerate(list_of_names[6:]):
    df = all_df[name]
    
    # Initial and maximum stress values and corresponding displacements
    initial_stress = min(df["Stress (kPa)"])
    initial_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == initial_stress].iloc[0]

    maximum_stress = max(df["Stress (kPa)"])
    maximum_displacement = df["Horizontal Displacement (mm)"][df["Stress (kPa)"] == maximum_stress].iloc[0]

    # Calculate the slope of the line
    slope = (maximum_displacement - initial_displacement) / (maximum_stress - initial_stress)

    # Plot the data on the second subplot
    axs[1].plot(df["Horizontal Displacement (mm)"], df["Stress (kPa)"], label=f'{name} Data')

    # Plot the line from initial stress to maximum stress
    axs[1].plot([initial_displacement, maximum_displacement], [initial_stress, maximum_stress], 'ro-', label=f'{name} Stress Displacement Line')

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
