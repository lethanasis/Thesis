# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:15:55 2024

@author: thana
"""

import numpy as np
from numpy.linalg import solve
import sys 
import scipy.constants
from IonHandler import IonHandler
import ionrate
from DistributionFunction import DistributionFunction
import h5py
from scipy.linalg import solve
from Radiation_losses import RadiationLosses, Transport 
from get_derivatives import evaluateBraamsConductivity, getEc, braams_conductivity_derivative_with_respect_to_Z, getCoulombLogarithm, derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature, derivative_sigma_T
from rewriting_matrices import construct_Jacobian, construct_F, Zeff


def power_balance(Te, ne, ions : IonHandler, Di, dist, NRE):
    Z = Zeff(ions, ne)
    Ec = getEc(Te, ne)
    e = scipy.constants.e
    c = scipy.constants.c
    sigma = evaluateBraamsConductivity(ne, Te, Z)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te)
    Ptransp, Ptransp_prime = Transport(ions, Di, dist, Te, Tw=0.025)
    
    F = e*c*NRE*Ec + sigma * Ec**2 - Prad - Ptransp
    return F 

def bisection(power_balance, a, b, ions, pn, Te, Tn, fre, Di, dist, NRE, tol = np.sqrt(np.finfo(np.float64).eps), max_iter = 100 ):
    c = (a+b)/2
    ne, n = ionrate.equilibriumAtPressure(ions, pn, c, c*scipy.constants.e, fre)
    ions.setSolution(n)
    
    fa = power_balance(a, ne, ions, Di, dist, NRE)
    fb = power_balance(b, ne, ions, Di, dist, NRE)
    if fa*fb>0:
        print("Error : The function hsa the same signs at the endpoints")
        return None
    iter_count = 0
    while iter_count < max_iter:
        c = (a+b)/2
        fc = power_balance(c, ne, ions, Di, dist, NRE)
        print(c)
        if abs(fc)<tol*c:
            return c #root within tolerance
        elif fa*fc<0:
            b=c # Root in the left half
            fb=fc
        else:
            a=c # Root in the right half
            fa=fc 
        iter_count+=1 
        ne, n = ionrate.equilibriumAtPressure(ions, pn, c, c*scipy.constants.e, fre)
        ions.setSolution(n)
    print('Max iterations reached')
    return None

def objective_function(x,ions,Z):
    """Scalar objective function representing squared norm of vector-valued function F."""
    n = x[:-2]
    ions.setSolution(n)
    ne = x[-2]
    Te = x[-1]
    F = construct_F(ions, ne, Te, Z)
    return 0.5 * np.linalg.norm(F)**2

def compute_gradient(x,ions,Z):
    n = x[:-2]
    ions.setSolution(n)
    ne = x[-2]
    Te = x[-1]
    J = construct_Jacobian(ions, ne, Te, Z)
    F = construct_F(ions, ne, Te, Z)
    return J.T @ F
    

def line_search(objective_function, compute_gradient, x, dx,ions, Z):
    alpha = 1.0
    c = 0.5
    rho = 0.5 
    while objective_function(x + alpha * dx,ions,Z) > objective_function(x,ions,Z) + c * alpha * compute_gradient(x,ions,Z).T@dx:
        alpha *= rho
    return alpha

def newton_method(ions: IonHandler, pn, ne, Te, fre, Z, tol = np.sqrt(np.finfo(np.float64).eps), max_iter=100):
    nfree, n = ionrate.equilibriumAtPressure(ions, pn, Te, Te*scipy.constants.e, fre)
    
    for i in range(max_iter):
        J = construct_Jacobian(ions, ne, Te, Z)
        F = construct_F(ions, ne, Te, Z)
        dx = np.linalg.solve(J, -F)
        
        
        alpha = line_search(objective_function, compute_gradient, np.hstack((n, ne, Te)), dx,ions, Z)
        ne += alpha * dx[-2]
        Te += alpha * dx[-1]
        n += alpha * dx[:-2]
        
        # Update ion densities
        ions.setSolution(n)
        
        if np.linalg.norm(dx[:-1]) <tol + tol * ne and np.linalg.norm(dx[-1]) <tol + tol * Te:
            break
    
    return n, ne, Te, i