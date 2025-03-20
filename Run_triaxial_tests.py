# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:43:08 2025
@author: Dylan van Bezooijen
Function: Run all tests 
"""

from Shearing_CID import Shearing_phase_CID
from Shearing_CIU import Shearing_phase_CIU
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import numpy as np
from Consolidation import consolidation_phase


#DRAINED TESTS
drained_tests = ['188','191','192']
drained_start_row = [9,7,9]
drained_value_column = [7,6,6]
drained_skiprow = [30,28,30]

#CONSOLIDATION PHASE 
cons_start_row = [9,9,9]
cons_value_column = [6,6,6]
cons_skiprow = [30,30,30]

#UNDRAINED TESTS
undrained_tests = ['0193','0194','0195']
undrained_start_row = [9,9,9]
undrained_value_column = [6,6,6]
undrained_skiprow = [30,30,30]
  
#prepare to store data
q_f_d = {}
p_f_d = {}
q_f_u = {}
p_f_u = {}
G0s = {}
G2s = {}
v_fs = {}
lambdas = {}
kappas = {}
test_parameters = {}

#%% RUN DATA ANALYSIS DRAINED TEST
for i,test in enumerate(drained_tests):
    G0_modulus,G2_modulus,q_f,p_f,parameters = Shearing_phase_CID(
        test, drained_start_row[i], drained_value_column[i], drained_skiprow[i])
    q_f_d[test] = float(q_f)
    p_f_d[test] = float(p_f)
    G0s[test] = float(G0_modulus)
    G2s[test] = float(G2_modulus)
    test_parameters[test] = parameters
    
    #Consolidation phase
    lambd,kappa = consolidation_phase(
        test, cons_start_row[i], cons_value_column[i], cons_skiprow[i])
    lambdas[test] = float(lambd)
    kappas[test] = float(kappa)

 #%% RUN DATA ANALYSIS UNDRAINED TEST
for i,test in enumerate(undrained_tests):
    G0_modulus,G2_modulus,q_f,p_f,parameters = Shearing_phase_CIU(
        test, undrained_start_row[i], undrained_value_column[i], undrained_skiprow[i])
    q_f_u[test] = float(q_f)
    p_f_u[test] = float(p_f)
    G0s[test] = float(G0_modulus)
    G2s[test] = float(G2_modulus)
    test_parameters[test] = parameters

# Extract values from dictionaries and convert to lists
p_f_values_drained = list(p_f_d.values())  # Mean stress values
q_f_values_drained = list(q_f_d.values())  # Deviatoric stress values

# Extract values from dictionaries and convert to lists
p_f_values_undrained = list(p_f_u.values())  # Mean stress values
q_f_values_undrained = list(q_f_u.values())  # Deviatoric stress values

#%% FUNCTION TO COMPUTE CRITICAL SLOPE
def critical_slope(p_f_values,q_f_values):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(p_f_values, q_f_values)
    
    # Critical state slope M is the regression slope
    M = slope
    
    # Compute critical state friction angle (phi_cs) from M
    phi_cs = np.arcsin(3 * M / (6 + M)) * (180 / np.pi)  # Convert to degrees
    
    # Print results
    print(f'The critical state slope (M) is {M:.3f}')
    print(f'The critical state friction angle (phi_cs) is {phi_cs:.3f} degrees')
    
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.regplot(x=p_f_values, y=q_f_values, scatter_kws={"color": "b"}, line_kws={"color": "r", "linestyle": "-"},ci=None)
    
    # Add annotation for the friction angle
    plt.text(
        x=max(p_f_values) * 0.6,  # Positioning the text around 60% of max p_f
        y=max(q_f_values) * 0.8,  # Positioning the text around 80% of max q_f
        s=f'ϕ_cs = {phi_cs:.2f}°',
        fontsize=12,
        color='red',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3')
    )
    
    plt.xlabel("p' (kPa)")
    plt.ylabel("q' (kPa)")
    plt.title('Failure Envelope of Triaxial Tests')
    
    # Grid and show plot
    plt.grid(True)
    plt.show()

#%% COMPUTE CRITICAL SLOPE
critical_slope(p_f_values_drained,q_f_values_drained)
critical_slope(p_f_values_undrained,q_f_values_undrained)
