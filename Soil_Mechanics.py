import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import linregress

# File path for the triaxial test data
file_path = r"Triaxial CID\Tx_188 CID.xls"


################# DATA CONSTANTS ##########################
# Read data from Excel file
raw_data = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None)

# Dictionary to store extracted test parameters
test_parameters = {}

# Extract relevant parameters from the dataset
for row in range(9, 20):
    parameter_name = raw_data.iloc[row, 3]     # Column D
    parameter_value = raw_data.iloc[row, 7]    # Column H
    test_parameters[parameter_name] = parameter_value
for row in range(9, 13):
    parameter_name = raw_data.iloc[row, 9]     # Column J
    parameter_value = raw_data.iloc[row, 10]   # Column K
    test_parameters[parameter_name] = parameter_value

# Only Specific parameters needed
dry_weight_grams = float(test_parameters.get("dry weight [g]:", "Not found"))
initial_weight_grams = float(test_parameters.get("initial weight [g]:", "Not found"))
soil_unit_weight_kN_per_m3 = float(test_parameters.get("g [kN/m3]:", "Not found"))
dry_soil_unit_weight_kN_per_m3 = float(test_parameters.get("gd [kN/m3]:", "Not found"))
initial_moisture_content = float(test_parameters.get("w0 [%]:", "Not found")) / 100  # Convert percentage to decimal
initial_void_ratio = float(test_parameters.get("e0 [-]:", "Not found"))
specific_gravity_soil = float(test_parameters.get("Gs [-]:", "Not found"))


################# DATA TOTAL ##########################
###### Read file
df = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None, usecols="A:Q", skiprows=30)

# Assign proper column names
column_names = ['Date_and_time', 'axial_total_stress_kPa', 'pore_pressure_kPa', 'radial_total_stress_kPa', 
                'axial_strain', 'volumetric_strain', 'kaman', 'temperature', 'D_Time', 'interval', 
                'D_pore_pressure', 'D_Height', 'Height', 'D_Volume', 'Volume', 'Area', 'Radius']
df.columns = column_names

# Drop unnecessary columns
df = df.drop(columns=['kaman', 'temperature'])

################# CSSM CALCULATIONS ##########################
# Effective stress calculations
df["radial_effective_stress_kPa"] = df["radial_total_stress_kPa"] - df["pore_pressure_kPa"]
df["axial_effective_stress_kPa"] = df["axial_total_stress_kPa"] - df["pore_pressure_kPa"]
df["mean_effective_stress_kPa"] = (df["axial_effective_stress_kPa"] + 2 * df["radial_effective_stress_kPa"]) / 3        #p' = (sigma(1) + 2*sigma(2) )/3
df["deviatoric_stress_kPa"] = df["axial_effective_stress_kPa"] - df["radial_effective_stress_kPa"]

# Calculate initial specific volume 
grain_density_kN_m3 = specific_gravity_soil * 9.81  
void_ratio_from_unit_weight = (grain_density_kN_m3 / dry_soil_unit_weight_kN_per_m3) - 1           # Void ratio already given. 
initial_specific_volume = 1 + void_ratio_from_unit_weight

# Calculate initial effective stress (p'0)
initial_effective_stress_kPa = df["mean_effective_stress_kPa"].iloc[0]      

# Overconsolidation Ratio (OCR)
overconsolidation_ratio = df["axial_total_stress_kPa"].iloc[0] / initial_effective_stress_kPa  # Needs verification

# M - Critical State Line Slope (q/p')
critical_state_slope_M = df["deviatoric_stress_kPa"].iloc[-1] / df["mean_effective_stress_kPa"].iloc[-1]  # Assuming last value is at critical state

# Fit Normal Compression Line (NCL)
log_mean_effective_stress = np.log(df["mean_effective_stress_kPa"])
specific_volume_array = df['Volume']        # np.full(len(df["mean_effective_stress_kPa"]), initial_specific_volume)

slope_lambda, intercept_gamma, _, _, _ = linregress(log_mean_effective_stress, specific_volume_array)  # λ is the slope


# Display Results
print(f"Initial Effective Stress (p'0): {initial_effective_stress_kPa:.3f} kPa")
print(f"Initial Specific Volume (v0): {initial_specific_volume:.3f}")
print(f"Critical State Parameter M: {critical_state_slope_M:.3f}")
print(f"Lambda (λ): {slope_lambda:.3f}")
print(f"Gamma (Γ): {intercept_gamma:.3f}")
print(f"Overconsolidation Ratio (OCR): {overconsolidation_ratio:.3f}")


################# Young's Modulus Calculations ##########################
# Find the linear portion of the curve (first 10 points)
linear_region = range(0, 10)
poisson_ratio = 0.3  # Assumed based on plasticity of 9.8%

#Young’s modulus: Axial Total Stress vs. Axial Strain
Youngs = np.polyfit(df["axial_strain"][linear_region], df["axial_total_stress_kPa"][linear_region], 1)[0] / 1e3  # Convert kPa to MPa

# Effective Young’s modulus: Axial Effective Stress vs. Axial Strain
Youngs_effective = np.polyfit(df["axial_strain"][linear_region], df["axial_effective_stress_kPa"][linear_region], 1)[0] / 1e3  # Convert kPa to MPa

# Shear Stiffness (G) from Young's modulus
Shear_Stiffness = Youngs / (2 * (1 + poisson_ratio))

# Print results
print(f"Total Young's Modulus (E): {Youngs:.3f} MPa")
print(f"Effective Young's Modulus (E'): {Youngs_effective:.3f} MPa")
print(f"Shear Stiffness (G): {Shear_Stiffness:.3f} MPa")


################# Plotting ##########################
# Plot subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Deviatoric Stress vs. Axial Strain
sns.lineplot(x=df["axial_strain"], y=df["deviatoric_stress_kPa"], ax=axes[0, 0])
axes[0, 0].set_title("Deviatoric Stress vs. Axial Strain")
axes[0, 0].set_xlabel("Axial Strain [-]")
axes[0, 0].set_ylabel("Deviatoric Stress [kPa]")

# Volumetric Strain vs. Deviatoric Strain
sns.lineplot(x=df["deviatoric_stress_kPa"], y=df["volumetric_strain"], ax=axes[0, 1])
axes[0, 1].set_title("Volumetric Strain vs. Deviatoric Stress")
axes[0, 1].set_xlabel("Deviatoric Stress [kPa]")
axes[0, 1].set_ylabel("Volumetric Strain [-]")

# Pore Pressure vs. Deviatoric Strain
sns.lineplot(x=df["deviatoric_stress_kPa"], y=df["pore_pressure_kPa"], ax=axes[1, 0])
axes[1, 0].set_title("Pore Pressure vs. Deviatoric Stress")
axes[1, 0].set_xlabel("Deviatoric Stress [kPa]")
axes[1, 0].set_ylabel("Pore Pressure [kPa]")

# CSSM Plot (Mean Effective Stress vs. Specific Volume)
sns.lineplot(x=df["mean_effective_stress_kPa"], y=specific_volume_array, ax=axes[1, 1])
axes[1, 1].set_title("CSSM Plot")
axes[1, 1].set_xlabel("Mean Effective Stress [kPa]")
axes[1, 1].set_ylabel("Specific Volume [-]")

plt.tight_layout()
plt.show()


# Plotting CSSM
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. v vs. log(p')
sns.lineplot(x=log_mean_effective_stress, y=specific_volume_array, ax=axes[0])
axes[0].set_title("v vs. log(p')")
axes[0].set_xlabel("log(p') [kPa]")
axes[0].set_ylabel("Specific Volume [-]")

# 2. v vs. p'
sns.lineplot(x=df["mean_effective_stress_kPa"], y=specific_volume_array, ax=axes[1])
axes[1].set_title("v vs. p'")
axes[1].set_xlabel("p' [kPa]")
axes[1].set_ylabel("Specific Volume [-]")

# 3. q vs. p'
sns.lineplot(x=df["mean_effective_stress_kPa"], y=df["deviatoric_stress_kPa"], ax=axes[2])
axes[2].set_title("q vs. p'")
axes[2].set_xlabel("p' [kPa]")
axes[2].set_ylabel("Deviatoric Stress [kPa]")

plt.tight_layout()
plt.show()