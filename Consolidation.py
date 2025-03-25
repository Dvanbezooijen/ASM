import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.stats import linregress



# Function for the consolidation phase
def consolidation_phase(test, start_row, value_column, skip_row):

    #%% LOAD DATA
    # File path for the triaxial test data
    file_path = f"Triaxial CID\\Tx_{test} CID.xls"
    
    # Read data from triaxial test
    raw_data = pd.read_excel(file_path, engine="xlrd", sheet_name="Consolidation", header=None)
    
    #%% LOAD CONSTANT PARAMETERS
    # Dictionary to store extracted test parameters
    test_parameters = {}
    
    # Set parameter names
    param_names = ['H_0', 'D_0', 'V_0', 'weight_0', 'weight_f', 'weight_dry', 'density',
                   'density_dry', 'w_0', 'G_s', 'e_0']
    
    # Set start row for triaxial data
    start_row = 9
    value_column = 6
    
    # Extract relevant parameters from the dataset
    for i, row in enumerate(range(start_row, start_row + 11)):
        parameter_value = raw_data.iloc[row, value_column]
        test_parameters[param_names[i]] = parameter_value
    
    #%% LOAD TEST DATA
    # Read table file
    df = pd.read_excel(file_path, sheet_name="Consolidation", header=None, usecols="A:Q", skiprows=skip_row, nrows=len(raw_data))
    
    # Define proper column names
    column_names = ['Date_and_time', 'axial_total_stress_kPa', 'pore_pressure_kPa', 'radial_total_stress_kPa', 
                    'axial_strain', 'volumetric_strain', 'kaman', 'temperature', 'D_Time', 'interval', 
                    'D_pore_pressure', 'D_Height', 'Height', 'D_Volume', 'Volume', 'Area', 'Radius']
    
    # Set column names
    df.columns = column_names
    
    #%% 1. PLOT CONSOLIDATION PLOT
    
    # Calculate stresses 
    df["radial_effective_stress_kPa"] = df["radial_total_stress_kPa"] - df["pore_pressure_kPa"]
    df["axial_effective_stress_kPa"] = df["axial_total_stress_kPa"] - df["pore_pressure_kPa"]
    df["deviatoric_stress_kPa"] = df["axial_effective_stress_kPa"] - df["radial_effective_stress_kPa"]
    
    # Calculate mean effective stress
    df["mean_effective_stress_kPa"] = (df["axial_effective_stress_kPa"] + 2 * df["radial_effective_stress_kPa"]) / 3
    df["log_mean_effective_stress"] = np.log10(df["mean_effective_stress_kPa"])
    
    # Calculate void ratio
    df["e"] = test_parameters['e_0'] - ((df['volumetric_strain'] / 100) / (1 + test_parameters['e_0']))
    
    #%% LAMBDA FIT
    # Select the last 200 points minus the last point (i.e., from -41 to -2 in the dataframe)
    linear_subset = df.iloc[-200:-1]
    
    # Perform linear regression on the selected subset
    slope, intercept, r_value, p_value, std_err = linregress(
        linear_subset["log_mean_effective_stress"], linear_subset["e"]
    )
    print(f"Slope (lambda): {slope}")
    
    #%% FIT KAPPA
    # Select last and first-to-last point to fit first kappa
    kappa = (df["e"].iloc[-1] - df["e"].iloc[-2]) / (df["log_mean_effective_stress"].iloc[-1] - df["log_mean_effective_stress"].iloc[-2])
    
    print(f"Slope (kappa): {kappa}")
    
    #%% PLOT FITTED LINES FOR KAPPA AND LAMBDA
    # Plot the original data
    plt.figure(figsize=(8, 6))
    plt.scatter(df["log_mean_effective_stress"], df["e"], color='b', label='Data', alpha=0.7)
    
    # Plot the fitted lambda line (using the last 40 points minus the last one)
    x_values_lambda = np.linspace(linear_subset["log_mean_effective_stress"].min(), linear_subset["log_mean_effective_stress"].max(), 100)
    y_values_lambda = intercept + slope * x_values_lambda
    plt.plot(x_values_lambda, y_values_lambda, color='r', label=f'Fitted Line (lambda)')

    # Plot the kappa line using the last two data points in the entire dataframe
    x_values_kappa = np.linspace(df["log_mean_effective_stress"].iloc[-2], df["log_mean_effective_stress"].iloc[-1], 100)
    y_values_kappa = df["e"].iloc[-2] + kappa * (x_values_kappa - df["log_mean_effective_stress"].iloc[-2])
    plt.plot(x_values_kappa, y_values_kappa, color='g', label=f'Fitted Line (kappa)')

    # Add labels and title
    plt.xlabel('Log of Mean Effective Stress (log10 kPa)')
    plt.ylabel('Void Ratio (e)')
    plt.title('Void Ratio vs Log of Mean Effective Stress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Return slope (lambda) and kappa
    return slope, kappa
