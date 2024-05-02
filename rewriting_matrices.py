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
Di = 1
dist = 0.25 * 1.2

def Zeff(ions:IonHandler, ne):
    x=0
    for ion in ions:
        for j in range(ion.Z+1):
            x += j**2 * ion.solution[j]
        Zeff = x/ne
        return Zeff
ne=1e19
Te=2
Z = Zeff(ions, ne)


def construct_Jacobian(ions: IonHandler, ne, Te, Z, fre=fre, V_plasma =1, V_vessel =1):
    N = ions.getNumberOfStates()+2
    J = np.zeros((N,N))
    e = scipy.constants.e
    eps0 = scipy.constants.epsilon_0
    c = scipy.constants.c
    pi = scipy.constants.pi
    m_e = scipy.constants.m_e
    Ec = getEc(Te, ne)
    #Derivatives of critical field
    dEc_dT, dEc2_dT, dEc_dne, dEc2_dne = getCriticalFieldDerivativeWithRespectToTemperature(Te, ne)
    dlogLambda_dn = get_derivatives.derivative_coulomb_log_n(Te, ne)
    logLambda = getCoulombLogarithm(Te, ne)
    
    sigma = evaluateBraamsConductivity(ne, Te, Z)
    dsigma = derivative_sigma_T(ne, Te, Z)  
    
    off = 0 
    '''Initialize values for power balance derivatives wrt to ni'''
    run_h_ni = []
    ohm_h_ni = []
    rad_l_ni = []
    neu_t_ni = []
    '''Initialize values for power balance derivatives wrt to ne'''
    run_h_ne = 0 
    ohm_h_ne = 0 
    rad_l_ne = 0 
    neu_t_ne = 0
    dZdne    = 0
    '''Initialize values for power balance derivatives wrt to Te'''
    run_h_Te = 0 
    ohm_h_Te = 0 
    rad_l_Te = 0 
    neu_t_Te = 0
    
    off = 0 
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
        
        '''Particle Balance'''
        for j in range(ion.Z+1):
            if j > 0:
                J[off+j,off+j-1] = I(j-1)*ne

            J[off+j,off+j] = -(I(j) + R(j))*ne

            if j < ion.Z:
                J[off+j,off+j+1] = R(j+1)*ne
            
            if j > 0 and j < ion.Z :
                J[off+j, N-2] += ne*(dIdne(j-1)*ion.solution[j-1] - dIdne(j)*ion.solution[j] + dRdne(j+1)*ion.solution[j+1] - dRdne(j)*ion.solution[j]) + \
                I(j-1) * ion.solution[j-1] - I(j) * ion.solution[j] + R(j+1)*ion.solution[j+1] - R(j)*ion.solution[j]
                J[off+j, N-1] += dIdT(j-1)*ne*ion.solution[j-1] - dIdT(j)*ne*ion.solution[j] + dRdT(j+1)*ne*ion.solution[j+1] - dRdT(j)*ne*ion.solution[j]
            elif j==0 :
                J[off+j, N-2] +=  ne*(-dIdne(j)*ion.solution[j] + dRdne(j+1)*ion.solution[j+1] - dRdne(j)*ion.solution[j]) - I(j)*ion.solution[j] + R(j+1)*ion.solution[j+1] - R(j)*ion.solution[j]
                J[off+j, N-1] += - dIdT(j)*ne*ion.solution[j] + dRdT(j+1)*ne*ion.solution[j+1] - dRdT(j)*ne*ion.solution[j]
                
            elif j==ion.Z:
                J[off+j, N-2] += ne*(dIdne(j-1)*ion.solution[j-1] - dIdne(j)*ion.solution[j] - dRdne(j)*ion.solution[j]) + I(j-1)*ion.solution[j-1] - I(j) * ion.solution[j] - R(j)*ion.solution[j]
                J[off+j, N-1] += dIdT(j-1)*ne*ion.solution[j-1] - dIdT(j)*ne*ion.solution[j] - dRdT(j)*ne*ion.solution[j]
            
            if fre is not None:
                if j < ion.Z:
                    J[off+j,off+j] -= ion.evaluateImpactIonizationRate(Z0=j, fre=fre)
                    if j > 0:
                        J[off+j,off+j-1] += ion.evaluateImpactIonizationRate(Z0=j-1, fre=fre)
        J[off+ion.Z,off:(off+ion.Z+1)] = 1
        J[off+ion.Z, off+ion.Z+1:]=0
        
        '''Total Electron Density'''
        for j in range(ion.Z+1):
            if j !=0 :
                J[N-2,off+j] = j
        J[N-2,N-2] = -1
        
        
        '''Power Balance derivatives wrt ion densities'''
        for j in range(ion.Z+1):
            ohm_h_ni.append(braams_conductivity_derivative_with_respect_to_Z(ne, Te, Z) * (j**2/ne) * getEc(Te, ne)**2)
            if j != ion.Z:
                rad_l_ni.append(I(j) * e * ion.IonThreshold[j] + L_line(j))
            else:
                rad_l_ni.append(0)
            if j==0:
                neu_t_ni.append((Di/(dist**2))*(Te-0.025)*e)
            else:
                neu_t_ni.append(0)


            J[N-1, off+j] = ohm_h_ni[off+j] - rad_l_ni[off+j]*ne - neu_t_ni[off+j]
        
        '''Power Balance derivative wrt electron density'''
        for j in range(ion.Z+1):
            dZdne -= j**2 * ion.solution[j]
            if j != ion.Z:
                rad_l_ne += (I(j) * e * ion.IonThreshold[j] + L_line(j))*ion.solution[j] + (L_free(j+1) - R(j+1)*e*ion.IonThreshold[j])*ion.solution[j+1] +\
                    ne*((dIdne(j)* e * ion.IonThreshold[j] + dL_linedne(j)) * ion.solution[j] + (dL_freedne(j+1) - dRdne(j+1) * e * ion.IonThreshold[j])*ion.solution[j+1])
        '''Power Balance derivative wrt electron temperature'''
        for j in range(ion.Z+1):
            if j == 0:
                neu_t_Te += (Di/(dist**2)) * e * ion.solution[0]
            if j != ion.Z:
                rad_l_Te += (dIdT(j)*e*ion.IonThreshold[j] + dL_linedT(j))*ion.solution[j] + (dL_freedT(j+1) - dRdT(j+1) * e * ion.IonThreshold[j])* ion.solution[j+1]
        off += ion.Z+1   

            
    dZdne = dZdne/(ne**2)
    run_h_ne = e * c * NRE * dEc_dne
    ohm_h_ne = braams_conductivity_derivative_with_respect_to_Z(ne, Te, Z)*dZdne * getEc(Te, ne)**2 + sigma * dEc2_dne    
    J[N-1, N-2] = run_h_ne + ohm_h_ne - rad_l_ne
    
    
    run_h_Te = e * c * NRE * dEc_dT
    ohm_h_Te = sigma * dEc2_dT + dsigma * getEc(Te, ne)**2
    J[N-1, N-1] = run_h_Te + ohm_h_Te - rad_l_Te*ne - neu_t_Te
    
    #test = sigma * e**6 * (2*ne) * getCoulombLogarithm(Te, ne) ** 2 /(16* pi**2 * eps0**4 * m_e**2 * c**4) + braams_conductivity_derivative_with_respect_to_Z(ne, Te, Z) * dZdne * getEc(Te, ne)**2
    #test = e**4 * NRE * logLambda / (4 * pi * eps0**2 * m_e *c) + e**4 * ne * NRE * dlogLambda_dn / (4* pi* eps0**2 * m_e * c)        
    #print(f'analytical {test}')
    #print(f'Runaway heating {run_h_ne}')
    #print(f'Ohmic heating {ohm_h_ne}')
    #print(f'Radiation losses {rad_l_ne}')
    
    return J

def construct_F(ions: IonHandler, ne, Te, Z, fre=fre):
    N = ions.getNumberOfStates()+2
    F = np.zeros((N,))
    Ec = getEc(Te, ne)
    e = scipy.constants.e
    eps0 = scipy.constants.epsilon_0
    c = scipy.constants.c
    pi = scipy.constants.pi
    m_e = scipy.constants.m_e
    Di = 1
    dist = 0.25*1.2
    sigma = evaluateBraamsConductivity(ne, Te, Z)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te)
    Ptransp, Ptransp_prime = Transport(ions, Di, dist, Te, Tw=0.025)    
    off = 0
    F[N-2] = -ne
    F[N-1] = e*c*NRE*Ec + sigma * Ec**2 - Prad - Ptransp
    
    for ion in ions: 
        I = lambda j :     0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j :     0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
        L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)    
        for j in range(ion.Z+1):
            if j == 0:
                F[off+j] = -(I(j)*ne + ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] + R(j+1)*ne*ion.solution[j+1] -R(j)*ne*ion.solution[j]
            elif j==ion.Z :
                F[off+j] = (I(j-1)*ne +ion.evaluateImpactIonizationRate(Z0=j-1,fre=fre))*ion.solution[j-1] - (I(j)*ne +ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] - R(j)*ne*ion.solution[j]
            else:
                F[off+j] = (I(j-1)*ne +ion.evaluateImpactIonizationRate(Z0=j-1,fre=fre))*ion.solution[j-1] - (I(j)*ne +ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] +R(j+1)*ne*ion.solution[j+1] - R(j)*ne*ion.solution[j]
            F[N-2] += j*ion.solution[j]
        if ion.name == 'D':
            F[1] = -ion.n
            for j in range(ion.Z+1):
                F[1] += ion.solution[j]
        if ion.name not in ['H', 'D']:
            F[N-3] = - ion.n
            x = 0 
            for j in range(ion.Z+1):
                x += ion.solution[j]
            F[N-3] = x - ion.n
        off += ion.Z+1
    return F
        

testJ = construct_Jacobian(ions, ne, Te, Z)
testF = construct_F(ions, ne, Te, Z)


''' Testing the power balance respond to temperatures'''
# =============================================================================
# 
# Te_arr = np.linspace(2.28,2.3,100)
# dat1 = []
# dat2 = []
# dat3 = []
# dat4 = []
# dat5 = []
# dat6=[]
# dat7=[]
# 
# e=scipy.constants.e
# c=scipy.constants.c
# for Te in Te_arr:
#     
#     ne=0
#     nfree, n = ionrate.equilibriumAtPressure(ions, 0.1, Te, Te*e, fre)
#     ions.setSolution(n)
#     for ion in ions:
#         for j in range(ion.Z+1):
#             ne += ion.solution[j]
# # =============================================================================
# #             if ion.name == 'Ne':
# #                 print(f'Deuterium density is {ion.n}')
# # =============================================================================
#     Prad, Prad_prime = RadiationLosses(ions, ne, Te)
#     Ptransp, Ptransp_prime = Transport(ions, Di, 0.25*1.2, Te, Tw=0.025)
#     print(f'Electron density is {ne}')
#     #print(n)
#     Ec = getEc(Te, ne)
#     sigma = evaluateBraamsConductivity(ne, Te, Z)
#     dat1.append(e*c*NRE*Ec)
#     dat2.append(Prad)
#     dat3.append(Ptransp)
#     dat4.append(sigma*(Ec**2))
#     dat5.append(e*c*NRE*Ec+sigma*(Ec**2) - Prad - Ptransp)
#     dat6.append(n[0])
#     dat7.append(n[2])
# dat1 = np.array(dat1)
# dat2 = np.array(dat2)
# dat3 = np.array(dat3)
# dat4 = np.array(dat4)
# dat5 = np.array(dat5)
# dat6 = np.array(dat6)
# dat7 = np.array(dat7)
# 
# #print(Prad)
# #print(test)
# #plt.plot(Te_arr, dat1,label='Heating')
# #plt.plot(Te_arr, dat2,label='losses')
# #plt.plot(Te_arr, dat3,label='Neutral transport')
# #plt.plot(Te_arr, dat4, label = 'Ohmic heating')
# plt.plot(Te_arr,dat5, linestyle = '--' ,label = 'Power Balance')
# # =============================================================================
# # plt.plot(Te_arr, dat6)
# # plt.plot(Te_arr, dat7)
# # =============================================================================
# plt.legend()
# #plt.yscale('symlog')
# plt.grid(True)
# #plt.yticks([-1e9, -1e6, -1e3, 0, 1e3, 1e6, 1e9])
# 
# =============================================================================

