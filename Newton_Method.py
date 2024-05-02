# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:05:25 2024

@author: thana
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import sys
import time
import scipy.constants
from IonHandler import IonHandler
import ionrate 
from DistributionFunction import DistributionFunction
import h5py

from scipy.linalg import solve

from Radiation_losses import RadiationLosses, Transport

from ITER import NRE
from get_derivatives import evaluateBraamsConductivity, getEc, braams_conductivity_derivative_with_respect_to_Z, getCoulombLogarithm, derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature, derivative_sigma_T

#from testing_matrices import construct_F, construct_matrix, Zeff
from rewriting_matrices import construct_Jacobian, construct_F, Zeff


sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM\py\DREAM\Formulas')

# Full path to the HDF5 file
filename = 'cache\data_ionization'

with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]

ions = IonHandler()
ions.addIon('D', 1, 1e19)
ions.addIon('Ne', 10, 1e19)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold  = INTNe

'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)
pn=0.1

nfree, n = ionrate.equilibriumAtPressure(ions, pn, 2, 2*scipy.constants.e, fre)
ions.setSolution(n)

Di = 1
dist = 0.25 * 1.2


ne=5e19
Te=2.2875

Z=Zeff(ions, ne)




def newton_method(ions: IonHandler, ne,Te, tol=1e-6, max_iter=1000):
    dx=[]
    #x=np.zeros((N,))
    nfree, n = ionrate.equilibriumAtPressure(ions, 0.1, Te, Te*scipy.constants.e, fre)
    count = 0
    for i in range(max_iter):
        J = construct_Jacobian(ions, ne, Te, Z)
        F = construct_F(ions, ne, Te, Z)
        dx=np.linalg.solve(J,-F)
# =============================================================================
#         if np.all(dx<tol):
#             return dx, iter
# =============================================================================
        if np.linalg.norm(dx[:-1]) < tol*ne:
            if np.linalg.norm(dx[-1])< tol*Te:
                break
        
        if np.isnan(dx).any():
            print(f'Nan values in array, stopping after {i+1} iterations ')
            break
        ne += dx[-2]
        Te += dx[-1]
        n += dx[:-2]
        ions.setSolution(n)
        
        #print(dx)
        #print(f'ne is {ne}')
        #print(f'nfree is {nfree}')
        #ions.setSolution(n)
        
        #print(Te)
    return n, ne, Te, i

#J = construct_matrix(ions, ne, Te, Z)
#F = construct_F(ions, ne, Te, Z)
# =============================================================================

# 
# dx = np.linalg.solve(J,-F)
# print(dx)
# =============================================================================


#A,b  = ionrate.construct_matrix(ions, ne, Te)
#dx = np.linalg.solve(J, -F)
#x=np.linalg.solve(A,b)
#print(x)
n, ne, Te, i= newton_method(ions,ne,Te)
print(f'Ion densities are {n}')
print(f'Electron density is {ne}')
print(f'Electron Temperature is {Te}')

print(f'iterations {i+1}')

# =============================================================================
# J = construct_Jacobian(ions, ne, Te, Z)
# F= construct_F(ions, ne, Te, Z)
# dx = np.linalg.solve(J,-F)
# print(dx)
# =============================================================================
