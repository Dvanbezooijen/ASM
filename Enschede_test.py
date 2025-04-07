import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd 
from Enschede_read_data import enschede_read_data
import os
import configparser

# List of names for the DataFrames
list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "7_630N", "8_810N", "9D_180N", "10D_720N", "11D_1260N"]
list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "7_630N", "8_810N"]
list_of_names = ["9D_180N", "10D_720N", "11D_1260N"]
all_df = enschede_read_data(list_of_names)

# Given effective stress in Newtons and yieldpoint in percentage
effective_normal_stress_N = {
    "1_180N": 180, "2_450N": 450, "4_990N": 990, "5_1260N": 1260,
    "6_720N": 720, "7_630N": 630, "8_810N": 810, "9D_180N": 180,
    "10D_720N": 720, "11D_1260N": 1260
}
yield_points = {
    "1_180N": 7.3, "2_450N": 9.8, "4_990N": 10, "5_1260N": 8.4,
    "6_720N": 8.5, "7_630N": 8.1, "8_810N": 10, "9D_180N": 11,
    "10D_720N": 12.5, "11D_1260N": 12
}


area_mm2 = 3600
area_m2 = area_mm2 * 1e-6  # Convert to m²
effective_normal_stress_kPa = {k: v / area_m2 * 1e-3 for k, v in effective_normal_stress_N.items()}

# Extract peak shear stress for each dataset
peak_shear_stress = {}
for name, df in all_df.items():
    peak_shear_stress[name] = df["Stress (kPa)"].max()  # Get max shear stress

# Convert to arrays for regression
sigma_prime = np.array([effective_normal_stress_kPa[name] for name in peak_shear_stress.keys()])
tau_peak = np.array(list(peak_shear_stress.values()))

# Perform linear regression (τ = σ' * tan(φ'))
slope, intercept, r_value, p_value, std_err = linregress(sigma_prime, tau_peak)

cohesion = intercept

print(f"The cohesion of the sand (c) is: {cohesion:.2f} kPa")
# Compute friction angle φ'
phi_prime = np.arctan(slope) * (180 / np.pi)  # Convert to degrees

print(f"Internal Angle of Friction (φ'): {phi_prime:.2f} degrees")

# Plot Mohr-Coulomb Failure Envelope
plt.figure(figsize=(6,5))
plt.scatter(sigma_prime, tau_peak, color='red', label="Data Points")
plt.plot(sigma_prime, slope * sigma_prime + intercept, label=f"Fit: φ' = {phi_prime:.2f}°", color='blue')
plt.xlabel("Effective Normal Stress (kPa)")
plt.ylabel("Shear Stress (kPa)")
plt.title("Mohr-Coulomb Failure Envelope")
plt.legend()
plt.grid()
plt.show()


## DILATION ANGLE
sample_heights = {}

# Read sample heights from .ini files
for sample_name in list_of_names:
    file_path = f"Data_Enschede/{sample_name}.ini"
    config = configparser.ConfigParser()
    if os.path.exists(file_path):
        config.read(file_path)
        try:
            height_mm = float(config["Physical Properties"]["Sample Height"])
            sample_heights[sample_name] = height_mm  # Store height
        except KeyError as e:
            print(f"Missing key {e} in file {file_path}, skipping.")
    else:
        print(f"File {file_path} not found, skipping.")

# Dictionary to store dilation angles
dilation_angles = {}

for name, df in all_df.items():
    if name in sample_heights:
        sample_height_mm = sample_heights[name]

        # Compute strain values
        df["Shear Strain"] = df["Horizontal Displacement (mm)"] / sample_height_mm
        df["Volumetric Strain"] = df["Vertical Displacement (mm)"] / sample_height_mm

        # Filter to use **only the failure phase**
        df_filtered = df[df["Shear Strain"] > 0.02]  # Keep data where shear strain > 2%

        # Perform regression for dilation angle (ψ = atan(slope))
        slope_dilation, _, _, _, _ = linregress(df_filtered["Shear Strain"], df_filtered["Volumetric Strain"])
        dilation_angle = np.arctan(slope_dilation) * (180 / np.pi)  # Convert to degrees
        dilation_angles[name] = dilation_angle

# Print results again
for name, psi in dilation_angles.items():
    print(f"Test {name}: Dilation Angle (ψ) = {psi:.2f} degrees")

# Calculate the average dilation angle for the entire soil
average_dilation_angle = np.mean(list(dilation_angles.values()))
print(f"Average Dilation Angle (ψ) for the soil: {average_dilation_angle:.2f} degrees")


# Extract the effective normal stress and dilation angles as lists
sigma_prime_list = [effective_normal_stress_kPa[name] for name in dilation_angles.keys()]
dilation_angles_list = list(dilation_angles.values())


## STRESS STRAIN CURVE ALL DATA.
plt.figure(figsize=(10, 8))

for name, df in all_df.items():
    # Ensure the data has the necessary columns, assuming "Shear Strain" and "Stress (kPa)" are already in the DataFrame
    if 'Shear Strain' in df.columns and 'Stress (kPa)' in df.columns:
        # Convert shear strain to percentage
        shear_strain_percentage = df["Shear Strain"] * 100
        plt.plot(shear_strain_percentage, df["Stress (kPa)"], label=name)
# Adding labels and title
plt.xlabel("Shear Strain (%)")
plt.ylabel("Shear Stress (kPa)")
plt.title("Stress-Strain Curves for All Datasets")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Dataset Names")
plt.grid(True)
plt.tight_layout()
plt.show()

######################  maximum plastic volumetric strain values
max_volumetric_strain_values = []

# Loop through datasets to calculate maximum plastic volumetric strain using yield points
for name, df in all_df.items():
    if name in yield_points:
        yield_shear_strain = yield_points[name] / 100  # Convert from percentage to decimal
        
        # Extract the post-yield data (shear strain > yield strain)
        post_yield_data = df[df["Shear Strain"] > yield_shear_strain]
        
        # Find the maximum volumetric strain in the post-yield region
        if len(post_yield_data) > 1:  # Ensure there is enough data for analysis
            max_volumetric_strain = post_yield_data["Volumetric Strain"].max()
            print(f"Test {name}: Maximum Plastic Volumetric Strain = {max_volumetric_strain:.4f}")
            max_volumetric_strain_values.append(max_volumetric_strain)
        else:
            print(f"Test {name}: Not enough post-yield data for analysis.")
            max_volumetric_strain_values.append(np.nan)  # Append NaN if there's not enough data
average_max_volumetric_strain = np.nanmean(max_volumetric_strain_values)
print(f"Average Maximum Plastic Volumetric Strain: {average_max_volumetric_strain:.4f}")

######################################################## Calculation of Aspect ratio for Mohr-Coulomb
phi_rad = np.radians(phi_prime)
# Compute theoretical aspect ratio from Mohr-Coulomb
R = (1 + np.sin(phi_rad)) / (1 - np.sin(phi_rad))
theoretical_aspect_ratio = (1 + R) / (R - 1)
print(f"\n--- Theoretical Ellipse Aspect Ratio from φ' = {phi_prime}°: {theoretical_aspect_ratio:.2f} ---\n")


######## Shear Modulus (G) and Youngs modlus (E) (from the linear region of the stress-strain curve)
shear_modulus = {}
youngs_modulus = {}
# Poisson's Ratio (v)
poissons_ratio = 0.275
# Loop through datasets to calculate Shear Modulus and Young's Modulus
for name, df in all_df.items():
    if 'Shear Strain' in df.columns and 'Stress (kPa)' in df.columns:
        # Extract the initial linear region (before the yield strain, assuming yield points)
        yield_shear_strain = yield_points.get(name, 0) / 100
        linear_region = df[df["Shear Strain"] <= yield_shear_strain]
        
        # Perform linear regression (Stress vs. Shear Strain) in the linear region
        if len(linear_region) > 1:  # Ensure enough data for regression
            slope, _, _, _, _ = linregress(linear_region["Shear Strain"], linear_region["Stress (kPa)"])
            shear_modulus[name] = slope  # Shear Modulus = Slope of Stress-Strain curve
            
            # Calculate Young's Modulus using the formula E = 2G(1 + v)
            youngs_modulus[name] = 2 * shear_modulus[name] * (1 + poissons_ratio)
            
            print(f"Test {name}: Shear Modulus (G) = {slope:.2f} kPa, Young's Modulus (E) = {youngs_modulus[name]:.2f} kPa")
        else:
            print(f"Test {name}: Not enough data for Shear Modulus calculation.")
            shear_modulus[name] = np.nan
            youngs_modulus[name] = np.nan

# Calculate the average Shear Modulus and Young's Modulus from all tests
average_shear_modulus = np.nanmean(list(shear_modulus.values()))
average_youngs_modulus = np.nanmean(list(youngs_modulus.values()))

print(f"Average Shear Modulus: {average_shear_modulus:.2f} kPa")
print(f"Average Young's Modulus: {average_youngs_modulus:.2f} kPa")



