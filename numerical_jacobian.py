import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import sys
import time
import scipy.constants
import ionrate
from IonHandler import IonHandler
import h5py
from DistributionFunction import DistributionFunction

from Radiation_losses import RadiationLosses, Transport
import get_derivatives
from ITER import NRE
from get_derivatives import evaluateBraamsConductivity, getEc, braams_conductivity_derivative_with_respect_to_Z, getCoulombLogarithm, derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature, derivative_sigma_T
import rewriting_matrices
from rewriting_matrices import Zeff, construct_F, testJ

sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM\py\DREAM\Formulas')

#Full parth to HDF5 file
filename = 'cache\data_ionization'

with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]
'''Set up Ions'''

ions = IonHandler()
ions.addIon('D', 1, 1e19)
ions.addIon('Ne', 10, 1e19)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold = INTNe

'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)
pn=0.1

nfree, n = ionrate.equilibriumAtPressure(ions, pn, 2, 2*scipy.constants.e, fre)

ions.setSolution(n)
Te = 2
ne = 1e19

Z = Zeff(ions, ne)

F = rewriting_matrices.construct_F(ions, ne, Te, Z)

N = ions.getNumberOfStates()+2
x = np.zeros((N,))
off=0

for ion in ions:
    for j in range(ion.Z+1):
        x[off+j] = ion.solution[j]
        if j ==ion.Z:
            x[off+j] = ion.n
    
    off+=ion.Z+1
x[N-2] = ne
x[N-1] = Te  

Fc = construct_F(ions, ne, Te, Z)


def numerical_jacobian(F, x):
    
    k = len(x)
    J = np.zeros((k,k))
    k = ions.getNumberOfStates()
    h = np.sqrt(np.finfo(np.float64).eps)
    l = len(x)
    for i in range(l):
        for j in range(l-2):
            n0 = n[j]
            n[j] = n[j] + h*ne
            ions.setSolution(n)
            J[i, j] = (construct_F(ions, ne, Te, Z)[i] - Fc[i]) / (h*ne)
            n[j] = n0
            
            
            
    for i in range(l):
        for j in range(l):
            neplus = ne + h*ne
            J[i,N-2] = (construct_F(ions, neplus, Te, Z)[i] - Fc[i])/ (h*ne)
    
    for i in range(l):
        for j in range(l):
            Teplus = Te + h*Te
            J[i,N-1] = (construct_F(ions, ne, Teplus, Z)[i] - Fc[i])/ (h*Te)
            
    
# =============================================================================
#     print(n)
#     for i in range(k):
#         n0 = n[i]
#         n[i] =  n[i] - n[i]
#         print(n)
#         n[i] = n0
# =============================================================================
        
               
    return J
J = numerical_jacobian(F, x)

diff = J-testJ

#print(J)