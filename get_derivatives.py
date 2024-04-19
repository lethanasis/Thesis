import numpy as np
import scipy.constants
import scipy.interpolate

def evaluateBraamsConductivity(n, T, Z):
    """
    Evaluates the conductivity formula by Braams and Karney at the
    requested temperature and effective plasma charge.
    """
    sigma = getBraamsConductivity()
    return sigma(n, T, Z)

def getBraamsConductivity():
    """
    Returns the conductivity for a relativistic plasma as calculated
    by Braams and Karney [Phys. Fluids B 1, 1355 (1989)].
    """
    mc2  = scipy.constants.physical_constants['electron mass energy equivalent in MeV'][0] * 1e6
    eps0 = scipy.constants.epsilon_0
    e    = scipy.constants.e
    me   = scipy.constants.m_e

    Tmc2  = np.array([0,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100])
    X     = np.array([0,0.090909090909091,0.166666666666667,0.333333333333333,0.5,1])
    sigma = np.array([12.7661,12.2972,11.8737,10.8120,9.5075,7.8269,5.4760,3.9694,2.8247,1.7887,1.2649,0.8944,0.5657,0.4000,11.3301,10.9587,10.6195,9.7540,8.6631,7.2156,5.1138,3.7221,2.6483,1.6738,1.1826,0.8359,0.5286,0.3738,10.3912,10.0778,9.7896,9.0462,8.0936,6.8043,4.8805,3.5730,2.5484,1.6116,1.1386,0.8047,0.5089,0.3598,8.7546,8.5328,8.3265,7.7844,7.0689,6.0624,4.4724,3.3261,2.3920,1.5180,1.0731,0.7585,0.4797,0.3392,7.4290,7.2736,7.1277,6.7381,6.2095,5.4367,4.1373,3.1347,2.2786,1.4538,1.0288,0.7274,0.4600,0.3253,3.7599,3.7549,3.7492,3.7285,3.6842,3.5713,3.1821,2.6501,2.0313,1.3301,0.9465,0.6704,0.4242,0.3000]).reshape(X.size, Tmc2.size)
    
    sbar  = scipy.interpolate.interp2d(Tmc2, X, sigma, kind='linear')
    normf = 4*np.pi*eps0**2 / (np.sqrt(me)*e**2)

    return lambda n, T, Z : sbar(T/mc2, 1/(1+Z)) * normf * (T*e)**(3/2) / (Z*getCoulombLogarithm(T, n))

def getCoulombLogarithm(T, n):
    """
    Calculates the Coulomb logarithm according to the formula given in
    Wesson's book "Tokamaks".

    :param float T: Plasma temperature (eV).
    :param float n: Plasma density (m^-3).
    """
    return 14.9 - 0.5*np.log(n / 1e20) + np.log(T / 1e3) 

def getEc(T, n):
    """
    Calculates the Connor-Hastie critical electric field, below which no
    runaway electrons can be generated.

    :param float T: Plasma temperature (eV).
    :param float n: Plasma density (m^-3).
    """
    logLambda = getCoulombLogarithm(T, n)

    c = scipy.constants.c
    e = scipy.constants.e
    me = scipy.constants.m_e
    eps0 = scipy.constants.epsilon_0

    return (n*logLambda*e**3) / (4*np.pi*eps0**2 * me * c**2)


def getConnorHastieCriticalField(T, n): return getEc(T, n)

"""Derivatives"""

# Define a function to calculate numerical derivatives
def numerical_derivative(func, x, h=1e-6):
    return (func(x + h) - func(x - h)) / (2 * h)

# Define a function to calculate the derivative of Braams conductivity with respect to temperature
def derivative_sigma_T(n, T, Z):
    conductivity_func = getBraamsConductivity()
    return numerical_derivative(lambda T: conductivity_func(n, T, Z), T)

# Define a function to calculate the derivative of Coulomb logarithm with respect to temperature
def derivative_coulomb_log_T(T, n):
    return numerical_derivative(lambda T: getCoulombLogarithm(T, n), T)

# Define a function to calculate the derivative of Coulomb logarithm with respect to electron density
def derivative_coulomb_log_n(T, n):
    return numerical_derivative(lambda n: getCoulombLogarithm(T, n), n)

def derivative_squared_coulomb_log_n(T, n):
    return numerical_derivative(lambda n: getCoulombLogarithm(T, n)**2, n)

def getCriticalFieldDerivativeWithRespectToTemperature(T, n):
    """
    Calculates the derivative of the Connor-Hastie critical electric field with respect to temperature.

    :param float T: Plasma temperature (eV).
    :param float n: Plasma density (m^-3).
    """
    logLambda = getCoulombLogarithm(T, n)

    c = scipy.constants.c
    e = scipy.constants.e
    me = scipy.constants.m_e
    eps0 = scipy.constants.epsilon_0

    dlnLambda_dT = derivative_coulomb_log_T(T, n)
    dlnLambda_dne = derivative_coulomb_log_n(T, n)

    Ec = (n * logLambda * e**3) / (4 * np.pi * eps0**2 * me * c**2)
    
    dEc_dT = Ec * dlnLambda_dT / logLambda

    dEc2_dT = 2 * Ec * dEc_dT
    
    dEc_dn = Ec * dlnLambda_dne / logLambda + Ec/n
    
    dEc2_dn = 2 * Ec * dEc_dn

    return dEc_dT, dEc2_dT, dEc_dn, dEc2_dn


def braams_conductivity_derivative_with_respect_to_Z(n, T, Z, h=1e-6):
    """
    Computes the derivative of Braams conductivity with respect to Z
    using numerical differentiation (central finite difference method).
    """
    conductivity_func = getBraamsConductivity()
    
    # Compute Braams conductivity at Z and Z+h
    conductivity_Z = conductivity_func(n, T, Z)
    conductivity_Z_h = conductivity_func(n, T, Z+h)
    
    # Compute central finite difference derivative
    derivative = (conductivity_Z_h - conductivity_Z) / h
    
    return derivative


