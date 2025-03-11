#%% IMPORT PACKAGES
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import linregress

#%% LOAD DATA
# File path for the triaxial test data
file_path = r"Triaxial CID\Tx_191 CID.xls"

# Read data from triaxial test
raw_data = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None)

#%% LOAD CONSTANT PARAMETERS

# Dictionary to store extracted test parameters
test_parameters = {}

# Set parameter names
param_names = ['H_0', 'D_0', 'V_0', 'weight_0', 'weight_f', 'weight_dry', 'density',
               'density_dry', 'w_0', 'G_s', 'e_0']

# Set start row for triaxial data
start_row = 7
value_column = 6

# Extract relevant parameters from the dataset
for i, row in enumerate(range(start_row, start_row + 11)):
    parameter_value = raw_data.iloc[row, value_column]
    test_parameters[param_names[i]] = parameter_value

#%% LOAD TEST DATA
#read table file
df = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None, usecols="A:Q", skiprows=30)

# Define proper column names
column_names = ['Date_and_time', 'axial_total_stress_kPa', 'pore_pressure_kPa', 'radial_total_stress_kPa', 
                'axial_strain', 'volumetric_strain', 'kaman', 'temperature', 'D_Time', 'interval', 
                'D_pore_pressure', 'D_Height', 'Height', 'D_Volume', 'Volume', 'Area', 'Radius']

# Set column names
df.columns = column_names

#%% CALCULATE STRESSES

# Calculate stresses
df["radial_effective_stress_kPa"] = df["radial_total_stress_kPa"] - df["pore_pressure_kPa"]
df["axial_effective_stress_kPa"] = df["axial_total_stress_kPa"] - df["pore_pressure_kPa"]
df["deviatoric_stress_kPa"] = df["axial_effective_stress_kPa"] - df["radial_effective_stress_kPa"]

# Calculate deviatoric strain
df['deviatoric_strain'] = (2/3) * (df["axial_strain"] - (df["volumetric_strain"] / 2))

#%% COMPUTE G0
# Function to calculate tangent modulus at a specific strain point
def calculate_tangent_modulus(strain, stress, target_strain):
    """Calculate the tangent modulus at a specific strain level using finite differences."""
    idx = np.abs(strain - target_strain).idxmin()
    
    if idx > 0 and idx < len(strain) - 1:
        tangent_modulus = (stress[idx + 1] - stress[idx - 1]) / (strain[idx + 1] - strain[idx - 1])
    else:
        tangent_modulus = np.nan  # Handle edge cases
    
    return tangent_modulus, idx

# Input: Select strain points for calculations
target_strain_tangent = 0.01  # Tangent modulus calculation strain

# Calculate tangent modulus at the selected strain point
G0_modulus, idx_tangent = calculate_tangent_modulus(df['deviatoric_strain'], df['deviatoric_stress_kPa'], target_strain_tangent)

#%% COMPUTE G2

# Input: Select strain points for calculations
target_strain_secant = 4.5  # Secant modulus calculation strain

# Find the index for the secant strain point
idx_secant = np.abs(df['deviatoric_strain'] - target_strain_secant).idxmin()

# Calculate secant modulus at the selected secant strain point
q_secant = df['deviatoric_stress_kPa'][idx_secant]
G2_modulus = q_secant / df['deviatoric_strain'][idx_secant]

#%% COMPUTE G50

# Compute 50% shear failure point (given by secant)
q_50 = q_secant * 0.50

# Find strain corresponding to q_50
idx_g50 = (np.abs(df['deviatoric_stress_kPa'] - q_50)).idxmin()
strain_50 = df['deviatoric_strain'][idx_g50]

# Compute G50 secant modulus
G50_modulus = q_50 / strain_50

#%% PLOT STRESS STRAIN DIAGRAM WITH STIFNESSES

# Find the max strain and stress in the dataset
strain_max = df['deviatoric_strain'].max()
q_max = df['deviatoric_stress_kPa'].max()+20

# Plot the stress-strain curve
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['deviatoric_strain'], y=df['deviatoric_stress_kPa'], color='b', alpha=0.7, label="Data Points")

# Define the x-axis range up to strain_max
x_range = np.linspace(0, strain_max, 100)

# Plot the tangent line (G0) until strain_max
y_tangent = G0_modulus * (x_range - df['deviatoric_strain'][idx_tangent]) + df['deviatoric_stress_kPa'][idx_tangent]
plt.plot(x_range, y_tangent, 'r--', label=f'G0 Tangent: Slope = {G0_modulus:.2f} kPa')

# Plot the extended G2 secant line until strain_max
y_secant_extended = G2_modulus * x_range
plt.plot(x_range, y_secant_extended, 'g--', label=f'G2 Secant: Slope = {G2_modulus:.2f} kPa')

# Plot the extended G50 secant line until strain_max
y_g50_extended = G50_modulus * x_range
plt.plot(x_range, y_g50_extended, 'm--', label=f'G50 Secant: Slope = {G50_modulus:.2f} kPa')

# Set the x and y limits
plt.xlim(0, strain_max)
plt.ylim(0, q_max)

# Labels and title
plt.xlabel("Deviatoric Strain (εq)")
plt.ylabel("Deviatoric Stress (kPa)")
plt.title("Deviatoric Stress vs Deviatoric Strain")

# Grid and legend
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Show plot
plt.show()

#%% PLOT CRITICAL STATE LINE (CSL)

# Calculate mean effective stress
df["mean_effective_stress_kPa"] = (df["axial_effective_stress_kPa"] + 2 * df["radial_effective_stress_kPa"]) / 3

# Perform linear regression on q vs. p'
M, intercept, r_value, p_value, std_err = linregress(df["mean_effective_stress_kPa"], df["deviatoric_stress_kPa"])

# Convert M to critical state friction angle (degrees)
phi_cs = np.degrees(np.arctan(M / 3))
print(f'The value of the critical state friction angle is {phi_cs}')

# Create figure
plt.figure(figsize=(8,6))

# Scatter plot of q vs. p'
sns.scatterplot(x=df["mean_effective_stress_kPa"], y=df["deviatoric_stress_kPa"], color='b', alpha=0.7, label="Data Points")

# Generate values for the regression line
p_range = np.linspace(df["mean_effective_stress_kPa"].min(), df["mean_effective_stress_kPa"].max(), 100)
q_regression = M * p_range + intercept  # Regression equation

# Plot regression line (Critical State Line)
plt.plot(p_range, q_regression, 'r--', linewidth=2, label=f'CSL: q = {M:.3f} p\' + {intercept:.3f}')

# Labels and title
plt.xlabel("Mean Effective Stress, p' (kPa)")
plt.ylabel("Deviatoric Stress, q (kPa)")
plt.title("q vs. p' with Critical State Line")

# Legend and Grid
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.show()

#%% PLOT VOLUMETRIC STRAIN vs DEVIATORIC STRAIN

# Plot the scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(y=df['volumetric_strain'], x=df['deviatoric_strain'], color='b', alpha=0.7)

# Labels and title
plt.ylabel("Volumetric Strain")
plt.xlabel("Deviatoric Strain")
plt.title("Volumetric Strain vs Deviatoric Strain")

# Show plot
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

#%% PLOT PORE PRESSURE vs DEVIATORIC STRAIN 

# Create the plot with markers and lines
plt.figure(figsize=(8, 6))
plt.plot(df['deviatoric_strain'], df['pore_pressure_kPa'], color='b', linestyle='-', alpha=0.7)

# Labels and title
plt.xlabel("Deviatoric Strain")
plt.ylabel("Pore Pressure (kPa)")
plt.title("Pore Pressure vs Deviatoric Strain")

# Show grid and plot
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
