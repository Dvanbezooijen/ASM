# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:46:10 2025
@author: Dylan van Bezooijen
Extract parameters for Piazza Grande values
"""

#%% IMPORT PACKAGES
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import linregress

#%% LOAD DATA
# File path for the triaxial test data
file_path = r"Triaxial CID\Tx_191 CID.xls"

# Read data from Excel file
raw_data = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None)

#%% LOAD CONSTANT PARAMTERS

# Dictionary to store extracted test parameters
test_parameters = {}

# Extract relevant parameters from the dataset
for row in range(7, 18):
    parameter_name = raw_data.iloc[row, 3]     # Column D
    parameter_value = raw_data.iloc[row, 6]    # Column H
    test_parameters[parameter_name] = parameter_value
for row in range(7, 11):
    parameter_name = raw_data.iloc[row, 8]     # Column J
    parameter_value = raw_data.iloc[row, 9]   # Column K
    test_parameters[parameter_name] = parameter_value


#%% LOAD TEST DATA
#read table file
data = pd.read_excel(file_path, engine="xlrd", sheet_name="Data", header=None, usecols="A:Q", skiprows=30)

# Define proper column names
column_names = ['Date_and_time', 'axial_total_stress_kPa', 'pore_pressure_kPa', 'radial_total_stress_kPa', 
                'axial_strain', 'volumetric_strain', 'kaman', 'temperature', 'D_Time', 'interval', 
                'D_pore_pressure', 'D_Height', 'Height', 'D_Volume', 'Volume', 'Area', 'Radius']

# Set column names
data.columns = column_names

#%% CREATE RELEVENT PLOTS





