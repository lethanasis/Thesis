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

from matrices import construct_F, construct_matrix, Zeff


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


ne=1e19
Te=2

Z=Zeff(ions, ne)




def newton_method(ions: IonHandler, ne,Te, tol=1e-6, max_iter=1000):
    dx=[]
    N=ions.getNumberOfStates()+2
    x=np.zeros((N,))
    for iter in range(max_iter):
        J = construct_matrix(ions, ne, Te, Z)
        F = construct_F(ions, ne, Te, Z)
        dx=np.linalg.solve(J,-F)
        #print(dx)
        if np.all(dx<tol):
            return dx, iter
        x+=dx
        ions.setSolution(x[:-2])
        ne = x[-2]
        Te = x[-1] 
    return x, iter

J = construct_matrix(ions, ne, Te, Z)
F = construct_F(ions, ne, Te, Z)

dx = np.linalg.solve(J, -F)

print(dx)
root, iterations = newton_method(ions,ne,Te)
print(f'Root is {root}')
print(f'iterations {iterations+1}')

