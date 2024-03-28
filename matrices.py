# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:36:27 2024

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

#print(ions['D'].solution)
#print(ions['Ne'].solution)

ne=1e22
Te=2

def Zeff(ions: IonHandler, ne):
    x=0
    for ion in ions:
        for j in range(ion.Z+1):
            x += j**2 * ion.solution[j]
    Zeff = x/ne
    return Zeff

Z = Zeff(ions, ne)

def construct_matrix(ions: IonHandler, ne, Te, Z, fre=None, V_plasma=1, V_vessel=1):
    """
    Construct the matrix for the ion rate equation.
    """
    start_time = time.time()
    N = ions.getNumberOfStates() + 2
    A = np.zeros((N, N))
    b = np.zeros((N,))
    e = scipy.constants.e
    eps0 = scipy.constants.epsilon_0
    c = scipy.constants.c
    pi = scipy.constants.pi
    m_e = scipy.constants.m_e
    Ec = getEc(Te, nfree)
    #print(nfree)
    
    def Zeff(ions: IonHandler, ne):
        x=0
        for ion in ions:
            for j in range(ion.Z+1):
                x += j**2 * ion.solution[j]
        Zeff = x/ne
        return Zeff

    Z = Zeff(ions, ne)
    sigma = evaluateBraamsConductivity(nfree, Te, Z)
    

    #iVf = lambda j : (V_plasma / V_vessel) if j==0 else 1
    #Zeff = Zeff(ions, nfree)
    off = 0
    dEc, dEc2 = getCriticalFieldDerivativeWithRespectToTemperature(Te, nfree)
    dsigma = derivative_sigma_T(nfree, Te, Z)
    transp_deriv = 0
    for ion in ions:
        I = lambda j :     0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j :     0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
        L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)
        dIdT = lambda j :  0 if j==ion.Z else ion.scd.deriv_Te(Z0=j, n=ne, T=Te)
        dRdT = lambda j :  0 if j==0     else ion.acd.deriv_Te(Z0=j, n=ne, T=Te)
        dIdne = lambda j : 0 if j==ion.Z else ion.scd.deriv_ne(Z0=j, n=ne, T=Te)
        dRdne = lambda j : 0 if j==0     else ion.acd.deriv_ne(Z0=j, n=ne, T=Te)
        dL_linedne = lambda j : 0 if j==ion.Z else ion.plt.deriv_ne(Z0=j, n=ne, T=Te)
        dL_freedne = lambda j : 0 if j==0 else ion.prb.deriv_ne(Z0=j, n=ne, T=Te)
        dL_linedT = lambda j : 0 if j==ion.Z else ion.plt.deriv_Te(Z0=j, n=ne, T=Te)
        dL_freedT = lambda j :0 if j==0 else ion.prb.deriv_Te(Z0=j, n=ne, T=Te)
        

        for j in range(ion.Z+1):
            if j > 0:
                A[off+j,off+j-1] = I(j-1)*ne

            A[off+j,off+j] = -(I(j) + R(j))*ne

            if j < ion.Z:
                A[off+j,off+j+1] = R(j+1)*ne
            
            if j > 0 and j < ion.Z :
                A[off+j, N-2] += ne*(dIdne(j-1)*ion.solution[j-1] - dIdne(j)*ion.solution[j] + dRdne(j+1)*ion.solution[j+1] - dRdne(j)*ion.solution[j]) + I(j-1) * ion.solution[j-1] - I(j) * ion.solution[j] + R(j)*ion.solution[j] - R(j+1)*ion.solution[j]
                A[off+j, N-1] += dIdT(j-1)*ne*ion.solution[j-1] - dIdT(j)*ne*ion.solution[j] +dRdT(j+1)*ion.solution[j+1] -dRdT(j)*ion.solution[j]
            elif j==0 :
                A[off+j, N-2] += - ne*(dIdne(j)*ion.solution[j] + dRdne(j+1)*ion.solution[j+1] - dRdne(j)*ion.solution[j]) - I(j)*ion.solution[j+1] +R(j)*ion.solution[j+1]- R(j)*ion.solution[j]
                A[off+j, N-1] += - dIdT(j)*ne*ion.solution[j] +dRdT(j+1)*ne*ion.solution[j+1] -dRdT(j)*ne*ion.solution[j]
                #print(A[off,j])
                #A[j,N-2] = 10
            elif j==ion.Z:
                A[off+j, N-2] += ne*(dIdne(j-1)*ion.solution[j-1] - dIdne(j)*ion.solution[j] - dRdne(j)*ion.solution[j]) + I(j-1)*ion.solution[j-1] - I(j) * ion.solution[j] - R(j)*ion.solution[j]
                A[off+j, N-1] += dIdT(j-1)*ne*ion.solution[j-1] -dIdT(j)*ne*ion.solution[j] - dRdne(j)*ne*ion.solution[j]
                
            A[N-2, off+j] = j
            A[N-2, N-2] = -1
            A[N-2, N-1] = 0
            
            x = 1 / (1+Z)
            sum1 = L_line(j) + L_free(j) 
            if j != ion.Z :
                sum1 += ion.IonThreshold[j] * (I(j) - R(j+1))
            
            
            A[N-1, off+j] += braams_conductivity_derivative_with_respect_to_Z(nfree, T=Te, Z=Z) * (-x**2 * Z * (j**2)/2) 
            if j == 0 :
                A[N-1, off+j] += Di /(dist**2) * (Te-0.025)
            
            A[N-1, N-2] -= ion.solution[j] *( dL_linedne(j) + dL_freedne(j)) #+ ion.IonThreshold[j] * (dIdne(j) - dRdne(j)))
            A[N-1, N-1] -= ion.solution[j] * (dL_linedT(j) + dL_freedT(j)) #+ ion.IonThreshold[j] *(dIdT(j) -dRdT(j)))
            if j != ion.Z:
                A[N-1, N-2] -= e*ion.IonThreshold[j]*(dIdne(j) - dRdne(j+1))
                A[N-1, N-1] -= e*ion.IonThreshold[j]*(dIdT(j) - dRdT(j+1))
# Add fast-electron impact ionization
            if fre is not None:
                if j < ion.Z:
                    A[off+j,off+j] -= ion.evaluateImpactIonizationRate(Z0=j, fre=fre)
                    if j > 0:
                        A[off+j,off+j-1] += ion.evaluateImpactIonizationRate(Z0=j-1, fre=fre)
        
        off += ion.Z+1               
        #A[off+ion.Z,off:(off+ion.Z+1)] = 1
        #b[off+ion.Z] = ion.n
        transp_deriv += (Di/(dist**2))*ion.solution[0]*scipy.constants.e
    #print(dRdne(1))
    A[N-1, N-2] = A[N-1, N-2] * nfree + ((e**4 *NRE) / (4*pi * eps0**2 *m_e *c) )* getCoulombLogarithm(Te, nfree) \
    + braams_conductivity_derivative_with_respect_to_Z(nfree, Te, Z) * e**6 * ne**2 *getCoulombLogarithm(Te, nfree)**2 /(16 * pi**2 *eps0**4 *m_e**2 * c**4) + evaluateBraamsConductivity(nfree, Te, Z) * (e**6 * ne * getCoulombLogarithm(Te, nfree)**2)/(16*pi**2 * eps0**4 * m_e**2 * c**4)
    #print(f'first term is {nfree*A[N-1, N-1]}')
    #print(f'second term is {e*c*NRE*dEc}')
    #print(f'third term is {dsigma * Ec**2}')
    #print(f' fourth term is {sigma * dEc2}')
    #print(f'trnasport term is {transp_deriv}')
    #print(dIdT(0))
    A[N-1, N-1] = A[N-1, N-1] * nfree + e*c*NRE*dEc + dsigma * Ec**2 + sigma * dEc2 - transp_deriv 
    sum1 = sum1* nfree
    offset=0
    for ion in ions:
        for j in range(ion.Z+1):
            A[N-1, offset+j] += sum1
    
    end_time = time.time()
    #print(f'Time elapsed {end_time - start_time}')
    return A



def construct_F(ions: IonHandler, ne, Te, Z, fre=None):
    N = ions.getNumberOfStates()+2
    F = np.zeros((N,))
    Ec = getEc(Te, nfree)
    e = scipy.constants.e
    eps0 = scipy.constants.epsilon_0
    c = scipy.constants.c
    pi = scipy.constants.pi
    m_e = scipy.constants.m_e
    Di = 1
    Dist = 0.25*1.2
    sigma = evaluateBraamsConductivity(nfree, Te, Z)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te, nfree)
    Ptransp, Ptransp_prime = Transport(ions, Di, Dist, Te, Tw=0.025)
    off=0
    for ion in ions:
        I = lambda j :     0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j :     0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
        L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)
        F[N-2] = ne
        F[N-1] = e*c*NRE*Ec +sigma*(Ec**2) - Prad - Ptransp
        for j in range(ion.Z+1):
            if j == 0:
                F[off+j] = -(I(j)*ne + ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] + R(j+1)*ne*ion.solution[j+1] -R(j)*ne*ion.solution[j]
            elif j==ion.Z :
                F[off+j] = (I(j-1)*ne +ion.evaluateImpactIonizationRate(Z0=j-1,fre=fre))*ion.solution[j-1] - (I(j)*ne +ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] - R(j)*ne*ion.solution[j]
            else:
                F[off+j] = (I(j-1)*ne +ion.evaluateImpactIonizationRate(Z0=j-1,fre=fre))*ion.solution[j-1] - (I(j)*ne +ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] +R(j+1)*ne*ion.solution[j+1] - R(j)*ne*ion.solution[j]
            F[N-2] -= j*ion.solution[j] 
            F[N-1]
        off += ion.Z+1
        
    #print(F)
    return F


A = construct_matrix(ions, ne, Te, Z)
F = construct_F(ions, ne, Te, Z)

x=solve(A,-F)
#print(x)