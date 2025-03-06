import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 

file_path = r"Triaxial CID\Tx_188 CID.xls"
df = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None, usecols="A:Q", skiprows=30)

column_names = ['Date_and_time', 's_ax','u_[inf]','s_rad', 'e_ax', 'e_vol','kaman','temp','D_T','interval','D_u','D_H','H','D_V','V','A','R']
df.columns = column_names


deviatoric_stress = df['s_ax']  
axial_strain = df['e_ax']

# Find the linear portion of the curve (e.g., first 10 points)
linear_region = range(0, 10)

# Calculate the slope of the deviatoric stress vs. axial strain (this gives Young's Modulus)
E = np.polyfit(axial_strain[linear_region], deviatoric_stress[linear_region], 1)[0]

# Plot Deviatoric Stress vs. Deviatoric Strain
plt.figure(figsize=(10, 6))
plt.plot(axial_strain, deviatoric_stress, label='Deviatoric Stress vs. Axial Strain')
plt.xlabel('Axial Strain')
plt.ylabel('Deviatoric Stress [kPa]')
plt.title('Deviatoric Stress vs. Axial Strain')
plt.legend()
plt.grid(True)
plt.show()


# Plot Volumetric Strain vs. Deviatoric Strain
volumetric_strain = df['e_vol']  # Replace with actual column name for volumetric strain

plt.figure(figsize=(10, 6))
plt.plot(axial_strain, volumetric_strain, label='Volumetric Strain vs. Deviatoric Strain')
plt.xlabel('Deviatoric Strain')
plt.ylabel('Volumetric Strain [%]')
plt.title('Volumetric Strain vs. Deviatoric Strain')
plt.legend()
plt.grid(True)
plt.show()

# Plot Pore Pressure vs. Deviatoric Strain
pore_pressure = df['u_[inf]']  # Replace with actual column name for pore pressure

plt.figure(figsize=(10, 6))
plt.plot(axial_strain, pore_pressure, label='Pore Pressure vs. Deviatoric Strain')
plt.xlabel('Deviatoric Strain')
plt.ylabel('Pore Pressure [kPa]')
plt.title('Pore Pressure vs. Deviatoric Strain')
plt.legend()
plt.grid(True)
plt.show()


# Assuming 'df' is your DataFrame with columns like 's ax' (deviatoric stress) and 'e vol' (volumetric strain)
# Clean the data and filter out any unwanted NaN values
df.dropna(subset=['s_ax', 'e_vol'], inplace=True)

# Consolidation phase: In this example, we assume the consolidation phase occurs before deviatoric stress is applied.
# You may need to filter the data for the consolidation phase manually based on your dataset.

# Example: Consolidation phase (assuming it starts before 5 kPa deviatoric stress)
consolidation_data = df[df['s_ax'] < 5]

# Example: Shear phase (where deviatoric stress exceeds a threshold, like 5 kPa)
shear_data = df[df['s_ax'] >= 5]

# Plotting Consolidation and Shear Paths
plt.figure(figsize=(12, 8))

# Plot consolidation path (deviatoric stress vs. volumetric strain)
plt.plot(consolidation_data['e_vol'], consolidation_data['s_ax'], label="Consolidation Path", color='blue', marker='o')

# Plot shear path (deviatoric stress vs. volumetric strain)
plt.plot(shear_data['e_vol'], shear_data['s_ax'], label="Shear Path", color='red', marker='x')

# Adding labels and title
plt.xlabel('Volumetric Strain [%]')
plt.ylabel('Deviatoric Stress [kPa]')
plt.title('CSSM Path: Consolidation and Shear Stages')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
