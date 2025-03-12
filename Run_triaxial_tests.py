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
from Consolidation import consoldiation_phase


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
q_fs = {}
p_fs = {}
G0s = {}
G2s = {}
G50s ={}
V0s = {}
v_fs = {}
lambdas = {}
kappas = {}

#%% RUN DATA ANALYSIS TRIAXIAL TESTS (REQUIRES INPUT USER)
for i,test in enumerate(drained_tests):
    G0_modulus,G2_modulus,G50_modulus,q_f,p_f,V0,v_f = Shearing_phase_CID(
        test, drained_start_row[i], drained_value_column[i], drained_skiprow[i])
    q_fs[test] = float(q_f)
    p_fs[test] = float(p_f)
    G0s[test] = float(G0_modulus)
    G2s[test] = float(G2_modulus)
    G50s[test] = float(G50_modulus)
    V0s[test] = float(V0)
    v_fs[test] = v_f
    
    #Consolidation phase
    lambd,kappa = consoldiation_phase(
        test, cons_start_row[i], cons_value_column[i], cons_skiprow[i])
    lambdas[test] = float(lambd)
    kappas[test] = float(kappa)
    
for i,test in enumerate(undrained_tests):
    G0_modulus,G2_modulus,G50_modulus,q_f,p_f,V0,v_f = Shearing_phase_CIU(
        test, undrained_start_row[i], undrained_value_column[i], undrained_skiprow[i])
    q_fs[test] = float(q_f)
    p_fs[test] = float(p_f)
    G0s[test] = float(G0_modulus)
    G2s[test] = float(G2_modulus)
    G50s[test] = float(G50_modulus)
    V0s[test] = float(V0)
    v_fs[test] = v_f

#%% Compute M
# Extract values from dictionaries and convert to lists
p_f_values = list(p_fs.values())  # Mean stress values
q_f_values = list(q_fs.values())  # Deviatoric stress values

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
sns.regplot(x=p_f_values, y=q_f_values, scatter_kws={"color": "b"}, line_kws={"color": "r", "linestyle": "--"})

plt.xlabel('p_f (Mean Stress)')
plt.ylabel('q_f (Deviatoric Stress)')
plt.title('Stress envelope of triaxial tests')
plt.grid(True)
plt.show()
    