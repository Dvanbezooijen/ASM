import os
import configparser
import numpy as np

# List of sample names, excluding 3_720N
list_of_names = ["1_180N", "2_450N", "4_990N", "5_1260N", "6_720N", "7_630N","8_810N", "9D_180N", "10D_720N", "11D_1260N"]

# Constants
base_area_cm2 = 6 * 6  # 6x6cm

# Storage for results
bulk_densities = {}
normal_densities = []
d_densities = []

# Function to compute bulk density
def compute_bulk_density(weight_g, height_mm, base_area_cm2):
    if weight_g <= 0 or height_mm <= 0:
        return None  # Skip invalid calculations
    weight_kg = weight_g / 1000  # Convert g to kg
    volume_m3 = (base_area_cm2 / 10000) * (height_mm / 1000)  # Convert cm² to m² and mm to m
    return weight_kg / volume_m3

# Read each ini file
for sample_name in list_of_names:
    file_path = f"Data_Enschede\{sample_name}.ini"
    config = configparser.ConfigParser()
    if os.path.exists(file_path):
        config.read(file_path)
        # Extract sample height and weight
        try:
            height_mm = float(config["Physical Properties"]["Sample Height"])
            weight_g = float(config["Physical Properties"]["Sample Weight"])
            
            # Compute bulk density
            density = compute_bulk_density(weight_g, height_mm, base_area_cm2)
            bulk_densities[sample_name] = density

            # Categorize into normal or D-samples
            if "D" in sample_name:
                d_densities.append(density)
            else:
                normal_densities.append(density)

        except KeyError as e:
            print(f"Missing key {e} in file {file_path}, skipping.")

    else:
        print(f"File {file_path} not found, skipping.")

# Calculate means
mean_normal = np.mean(normal_densities) if normal_densities else None
mean_d = np.mean(d_densities) if d_densities else None
mean_all = np.mean(list(bulk_densities.values())) if bulk_densities else None

# Print results
print("\nIndividual Bulk Densities (kg/m³):")
for name, density in bulk_densities.items():
    print(f"{name}: {density:.2f} kg/m³")

print("\nAveraged Bulk Densities:")
print(f"Mean Normal Samples: {mean_normal:.2f} kg/m³" if mean_normal else "No normal samples found")
print(f"Mean D Samples: {mean_d:.2f} kg/m³" if mean_d else "No D samples found")
print(f"Mean All Samples: {mean_all:.2f} kg/m³" if mean_all else "No valid data found")



