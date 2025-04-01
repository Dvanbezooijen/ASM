import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd 
from Enschede_read_data import enschede_read_data
import os
import configparser

# List of names for the DataFrames
list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "7_630N", "8_810N", "9D_180N", "10D_720N", "11D_1260N"]
#list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "7_630N", "8_810N"]
#list_of_names = ["9D_180N", "10D_720N", "11D_1260N"]
all_df = enschede_read_data(list_of_names)

# Given effective stress in Newtons and yieldpoint in percentage
effective_normal_stress_N = {
    "1_180N": 180, "2_450N": 450, "4_990N": 990, "5_1260N": 1260,
    "6_720N": 720, "7_630N": 630, "8_810N": 810, "9D_180N": 180,
    "10D_720N": 720, "11D_1260N": 1260
}
yield_points = {
    "1_180N": 6.3, "2_450N": 9.8, "4_990N": 10, "5_1260N": 8.4,
    "6_720N": 8.5, "7_630N": 7.5, "8_810N": 8.1, "9D_180N": 10.6,
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
""" SHIT PLOT 
# Plot dilation angle vs effective normal stress
plt.figure(figsize=(8, 6))
plt.scatter(sigma_prime_list, dilation_angles_list, color='blue')
plt.xlabel('Effective Normal Stress (kPa)')
plt.ylabel('Dilation Angle (ψ) in Degrees')
plt.title('Dilation Angle vs. Confining Pressure')
plt.grid(True)
plt.show()
"""

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

# Show the plot
plt.show()

# ####################################################### isotropic hardening modulus values
hardening_modulus_values = []

# Loop through datasets to calculate isotropic hardening modulus using yield points
for name, df in all_df.items():
    if name in yield_points:
        yield_shear_strain = yield_points[name] / 100  # Convert from percentage to decimal
        
        # Extract the post-yield data (shear strain > yield strain)
        post_yield_data = df[df["Shear Strain"] > yield_shear_strain]
        
        # Perform linear regression (stress vs. shear strain) in the post-yield region
        if len(post_yield_data) > 1:  # Ensure there is enough data for regression
            slope, intercept, r_value, p_value, std_err = linregress(post_yield_data["Shear Strain"], post_yield_data["Stress (kPa)"])
            print(f"Test {name}: Isotropic Hardening Modulus = {slope:.2f} kPa")
            hardening_modulus_values.append(slope)
        else:
            print(f"Test {name}: Not enough post-yield data for regression.")
            hardening_modulus_values.append(np.nan)  # Append NaN if there's not enough data
            
average_hardening_modulus = np.nanmean(hardening_modulus_values)
print(f"Average Isotropic Hardening Modulus: {average_hardening_modulus:.2f} kPa")
# Plotting Isotropic Hardening Modulus for all tests
plt.figure(figsize=(8, 6))
test_names = list(yield_points.keys())

plt.bar(test_names, hardening_modulus_values, color='lightblue')
plt.xlabel("Test Name")
plt.ylabel("Isotropic Hardening Modulus (kPa)")
plt.title("Isotropic Hardening Modulus for Different Tests")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis='y')
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


# Plotting Maximum Plastic Volumetric Strain for all tests
plt.figure(figsize=(8, 6))
plt.bar(test_names, max_volumetric_strain_values, color='lightcoral')
plt.xlabel("Test Name")
plt.ylabel("Maximum Plastic Volumetric Strain")
plt.title("Maximum Plastic Volumetric Strain for Different Tests")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()



# 1. Ellipse Aspect Ratio
ellipse_aspect_ratio = {}

# Loop through datasets to calculate ellipse aspect ratio
for name, df in all_df.items():
    if name in yield_points:
        # Extract the post-yield data
        yield_shear_strain = yield_points[name] / 100  # Convert from percentage to decimal
        post_yield_data = df[df["Shear Strain"] > yield_shear_strain]
        
        # Calculate the principal stresses (assuming normal stresses are in the "Stress (kPa)" column)
        if len(post_yield_data) > 1:  # Ensure enough data points for calculation
            # Max and Min stress are assumed to be the maximum and minimum of the effective normal stress
            max_stress = post_yield_data["Stress (kPa)"].max()
            min_stress = post_yield_data["Stress (kPa)"].min()

            # Calculate shear stress as half the difference between max and min stresses
            shear_stress = (max_stress - min_stress) / 2

            # Calculate the ellipse aspect ratio
            aspect_ratio = (max_stress - min_stress) / (2 * shear_stress)  # Aspect ratio formula
            ellipse_aspect_ratio[name] = aspect_ratio
            print(f"Test {name}: Ellipse Aspect Ratio = {aspect_ratio:.2f}")
        else:
            print(f"Test {name}: Not enough data for ellipse aspect ratio calculation.")
            ellipse_aspect_ratio[name] = np.nan
average_ellipse_aspect_ratio = np.nanmean(list(ellipse_aspect_ratio.values()))
print(f"Average Ellipse Aspect Ratio: {average_ellipse_aspect_ratio:.2f}")
# Plotting Ellipse Aspect Ratio
plt.figure(figsize=(8, 6))
plt.bar(ellipse_aspect_ratio.keys(), ellipse_aspect_ratio.values(), color='lightgreen')
plt.xlabel("Test Name")
plt.ylabel("Ellipse Aspect Ratio")
plt.title("Ellipse Aspect Ratio for Different Tests")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


# 2. Young's Modulus (from the linear region of the stress-strain curve)
youngs_modulus = {}

# Loop through datasets to calculate Young's Modulus
for name, df in all_df.items():
    if 'Shear Strain' in df.columns and 'Stress (kPa)' in df.columns:
        # Extract the initial linear region (before the yield strain, assuming yield points)
        yield_shear_strain = yield_points.get(name, 0) / 100
        linear_region = df[df["Shear Strain"] <= yield_shear_strain]
        
        # Perform linear regression (Stress vs. Shear Strain) in the linear region
        if len(linear_region) > 1:  # Ensure enough data for regression
            slope, _, _, _, _ = linregress(linear_region["Shear Strain"], linear_region["Stress (kPa)"])
            youngs_modulus[name] = slope  # Young's Modulus = Slope of Stress-Strain curve
            print(f"Test {name}: Young's Modulus = {slope:.2f} kPa")
        else:
            print(f"Test {name}: Not enough data for Young's Modulus calculation.")
            youngs_modulus[name] = np.nan
average_youngs_modulus = np.nanmean(list(youngs_modulus.values()))
print(f"Average Young's Modulus: {average_youngs_modulus:.2f} kPa")
# Plotting Young's Modulus
plt.figure(figsize=(8, 6))
plt.bar(youngs_modulus.keys(), youngs_modulus.values(), color='lightblue')
plt.xlabel("Test Name")
plt.ylabel("Young's Modulus (kPa)")
plt.title("Young's Modulus for Different Tests")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
