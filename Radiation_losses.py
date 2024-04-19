# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:16:57 2024

@author: thana
"""

import  numpy as np
import scipy.constants
from IonHandler import IonHandler
from ADAS import data
import matplotlib.pyplot as plt
import sys
import h5py
import config
import ionrate
from DistributionFunction import DistributionFunction
from ITER import NRE


sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\particlebalance')
sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM')

# Full path to the HDF5 file
filename = 'cache\data_ionization'

# Get ionization threshold for all ions from the HDF5 file. 
with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]


def RadiationLosses(ions: IonHandler, ne, Te):
    
    retval = 0
    dretval = 0
    test=0
    
    for ion in ions:
        for j in range(ion.Z):
            I = lambda j : 0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
            R = lambda j : 0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
            L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
            L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)
            x = ( I(j) * (scipy.constants.e* ion.IonThreshold[j])  + L_line(j) ) * ion.solution[j] 
            y = (-R(j+1) * (scipy.constants.e * ion.IonThreshold[j] ) + L_free(j+1) ) *ion.solution[j+1]
            if j != ion.Z:
                test += (L_line(j) + L_free(j) + scipy.constants.e*ion.IonThreshold[j] * (I(j)-R(j))) * ion.solution[j]
            retval += x+y
            dI = lambda j : 0 if j==ion.Z else ion.scd.deriv_Te(Z0=j, n=ne, T=Te)
            dR = lambda j : 0 if j==0 else ion.acd.deriv_Te(Z0=j, n=ne, T=Te)
            dL_line = lambda j : 0 if j==ion.Z else ion.plt.deriv_Te(Z0=j, n=ne, T=Te)
            dL_free = lambda j : 0 if j==0 else ion.prb.deriv_Te(Z0=j, n=ne, T=Te)
            x_prime = ( dL_line(j) + dL_free(j) + (scipy.constants.e* ion.IonThreshold[j] )* (dI(j) - dR(j)) ) * ion.solution[j]
            dretval += x_prime
            tot_prime = dretval
            tot = retval
            
    test = test*ne
    #print(test)
    #print(tot*nfree)
    
    #print(f'Testing how I have it normally {test}')
    #print(f'Testing how it is in radiation losses {tot*nfree}')
    #print(tot_prime)
    #print(f"I_D = {I_arr_D}")
    #print(f"R_D = {R_arr}")
    #testing by replacing tot*nfree by the new test variable
    return tot*ne, tot_prime*ne

def Transport(ions : IonHandler, Di, Dist, Te, Tw):
    #Function that calculates and returns the transport losses. 
    retval = 0 
    for ion in ions:
        Ptransp = ( Di/(Dist**2) ) * scipy.constants.e *(Te - Tw) * ion.solution[0] 
        retval += Ptransp
        Ptransp_prime = ( Di/(Dist**2)) * ion.solution[0] * scipy.constants.e
    return Ptransp, Ptransp_prime

#ions.setSolution(n)


