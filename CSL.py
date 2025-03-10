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

#%% PLOT DEVIATORIC STRESS VS STRAIN

#calculate stresses 
df["radial_effective_stress_kPa"] = df["radial_total_stress_kPa"] - df["pore_pressure_kPa"]
df["axial_effective_stress_kPa"] = df["axial_total_stress_kPa"] - df["pore_pressure_kPa"]
df["deviatoric_stress_kPa"] = df["axial_effective_stress_kPa"] - df["radial_effective_stress_kPa"]

#calculate deviatoric strain
df['deviatoric_stress'] = (2/3)*(df["axial_strain"]-(df["volumetric_strain"]/2))

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['deviatoric_stress'], y=df['deviatoric_stress_kPa'], color='b', alpha=0.7)

# Labels and title
plt.xlabel("Deviatoric Strain (εq)")
plt.ylabel("Deviatoric Stress (kPa)")
plt.title("Deviatoric Stress vs Deviatoric Strain")

# Grid and show plot
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

#%% PLOT CRITICAL STATE LINE (CSL)

# Calculate mean effective stress
df["mean_effective_stress_kPa"] = (df["axial_effective_stress_kPa"] + 2 * df["radial_effective_stress_kPa"]) / 3

# Perform linear regression on q vs. p'
M, intercept, r_value, p_value, std_err = linregress(df["mean_effective_stress_kPa"], df["deviatoric_stress_kPa"])

# Convert M to critical state friction angle (degrees)
phi_cs = np.degrees(np.arctan(M / 3))
print(f'The value of the critical state friction is {phi_cs}')

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



