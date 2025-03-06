import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import linregress

file_path = r"Triaxial CID\Tx_188 CID.xls"


########################################### CONSTANTS ################
###### Read file
dt = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None)
# Dictionary to store the names and their corresponding values
data_dict = {}

# Loop through rows 9 to 19 (corresponding to D10 to D20)
for row in range(9, 20):
    name_in_d = dt.iloc[row, 3]     # Column D is index 3
    value_in_h = dt.iloc[row, 7]    # Column H is index 7
    data_dict[name_in_d] = value_in_h #Store
for row in range(9, 13):
    name_in_d = dt.iloc[row, 9]     # Column J is index 9
    value_in_h = dt.iloc[row, 10]   # Column K is index 10
    data_dict[name_in_d] = value_in_h #Store

dry_weight_value = float(data_dict.get("dry weight [g]:", "Not found"))         
initial_weight_value = float(data_dict.get("initial weight [g]:", "Not found"))
soil_unit_weight = float(data_dict.get("g [kN/m3]:","Not found"))
dry_soil_unit_weight = float(data_dict.get("gd [kN/m3]:","Not found"))
initial_moisture = float(data_dict.get("w0 [%]:","Not found"))/100      # Set it to decimals instead of percentages. 
initial_void_ratio = float(data_dict.get("e0 [-]:","Not found"))
specific_gravity = float(data_dict.get("Gs [-]:","Not found"))

############################################### DATA
###### Read file 
df = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None, usecols="A:Q", skiprows=30)

# Give each column correct name
column_names = ['Date_and_time', 'stress_ax','porepressure','stress_rad', 'strain_ax', 'strain_vol','kaman','temp','D_Time','interval','D_porepressure','D_Height','Height','D_Volume','Volume','Area','Radius']
df.columns = column_names
# Drop irrelevant columns
df = df.drop(columns=['kaman', 'temp'])     # Uncertain what kaman is. Temperature not needed. 

# INIT
deviatoric_stress = df['stress_ax']         #Deviatoric stress
axial_strain = df['strain_ax']              #Axial strain
pore_pressure = df['porepressure']          #Porepressure
volumetric_strain = df['strain_vol']        #Volumetric strain
poisson_ratio = 0.3                         # Assumed with a plastiticy of 9.8% at 0.3


## Basic calculations:
effective_stress = deviatoric_stress - pore_pressure  # Effective axial stress (axial stress minus pore pressure)

# Calculate void ratio using unit weights
gamma_s = specific_gravity * 9.81  
void_ratio_unit_weight = (gamma_s / dry_soil_unit_weight) - 1           # Void ratio already given. 

# Calculate initial specific volume 
specific_volume = 1 + void_ratio_unit_weight

# Calculate initial Effective Stress (p'0)
initial_effective_stress = deviatoric_stress[0] - pore_pressure[0]      

# Calculate Overconsolidation Ratio
OCR = deviatoric_stress[0]/initial_effective_stress             # denk niet dat dit klopt.... 


# Calculate mean effective stress (p')
df["stress_rad_effective"] = df["stress_rad"] - df["porepressure"]
df["stress_ax_effective"] = df["stress_ax"] - df["porepressure"]
df["p_mean"] = (df["stress_ax_effective"] + 2 * df["stress_rad_effective"]) / 3

# Deviatoric stress
df["deviatoric_stress"] = df["stress_ax_effective"] - df["stress_rad_effective"]

# M - Critical State Line Slope (q/p')
M = df["deviatoric_stress"].iloc[-1] / df["p_mean"].iloc[-1]  # Assuming last value is at critical state


# Fit Normal Compression Line (NCL)
ln_p = np.log(df["p_mean"])
specific_volume_series = np.full(len(df["p_mean"]), specific_volume)


slope, intercept, _, _, _ = linregress(ln_p, specific_volume_series)  # λ is the slope

λ = slope
Γ = intercept  # Extrapolated specific volume at p' = 1 kPa

# Display Results
print(f"Critical State Parameter M: {M:.3f}")
print(f"Lambda (λ): {λ:.3f}")
print(f"Gamma (Γ): {Γ:.3f}")

# Plot q vs p' (CSL)
plt.figure(figsize=(6, 4))
plt.plot(df["p_mean"], df["deviatoric_stress"], 'bo-', label="Stress Path")
plt.xlabel("Mean Effective Stress, p' (kPa)")
plt.ylabel("Deviatoric Stress, q (kPa)")
plt.title("Critical State Line (CSL)")
plt.legend()
plt.grid()
plt.show()

# Find the linear portion of the curve (e.g., first 10 points)
linear_region = range(0, 10)

# Calculate the slope of the deviatoric stress vs. axial strain (this gives Young's Modulus)
Youngs = np.polyfit(axial_strain[linear_region], deviatoric_stress[linear_region], 1)[0]             # Young's Modulus
Youngs_effective = np.polyfit(axial_strain[linear_region], effective_stress[linear_region], 1)[0]    # Effective Young's Modulus
Shear_Strength = Youngs / (2 * (1 + poisson_ratio))                                                  # Shear modulus formula

# Print calculated results
print(f"Total Young's Modulus (E): {Youngs} Pa")
print(f"Effective Young's Modulus (E'): {Youngs_effective} Pa")
print(f"Shear Stiffness (G): {Shear_Strength} Pa")


"""
# Fit a linear regression for e vs p' (Critical State Line)
lambda_slope, intercept = np.polyfit(effective_stress, void_ratio, 1)

# Fit a linear regression for q vs p' (Critical State Line)
M_slope, intercept = np.polyfit(effective_stress, deviatoric_stress, 1)

# Assuming friction angle (phi) for N (if not directly available)
phi = 30  # degrees
N = np.tan(np.radians(phi))  # N based on friction angle

# Output the results
print(f"Lambda (λ): {lambda_slope}")
print(f"M: {M_slope}")
print(f"N: {N}")

# Plot the CSL in e-p' and q-p' space
plt.figure(figsize=(12, 6))

# Plot e vs p'
plt.subplot(1, 2, 1)
plt.plot(effective_stress, void_ratio, 'o', label='Data')
plt.plot(effective_stress, np.polyval([lambda_slope, intercept], effective_stress), label=f"Fit: λ={lambda_slope:.3f}")
plt.xlabel('Effective Stress (p\')')
plt.ylabel('Void Ratio (e)')
plt.title('Void Ratio vs Effective Stress')
plt.legend()

# Plot q vs p'
plt.subplot(1, 2, 2)
plt.plot(effective_stress, deviatoric_stress, 'o', label='Data')
plt.plot(effective_stress, np.polyval([M_slope, intercept], effective_stress), label=f"Fit: M={M_slope:.3f}")
plt.xlabel('Effective Stress (p\')')
plt.ylabel('Deviatoric Stress (q)')
plt.title('Deviatoric Stress vs Effective Stress')
plt.legend()

plt.tight_layout()
plt.show()

"""
"""
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
plt.figure(figsize=(10, 6))
plt.plot(axial_strain, volumetric_strain, label='Volumetric Strain vs. Deviatoric Strain')
plt.xlabel('Deviatoric Strain')
plt.ylabel('Volumetric Strain [%]')
plt.title('Volumetric Strain vs. Deviatoric Strain')
plt.legend()
plt.grid(True)
plt.show()

# Plot Pore Pressure vs. Deviatoric Strain
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
"""