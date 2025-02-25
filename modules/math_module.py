import numpy as np
import mpmath
from scipy.optimize import fsolve

'''
#################################################
Constants
#################################################
'''

kB = 1.38064852e-23
e = 1.602176634e-19
h = 6.62607015e-34
hbar = h / (2*np.pi)
me = 9.1093837e-31
global lamb
lamb = 0

fermi1_10 = []
with open ("fermi_int_-10_20.txt","r") as file:
    for line in file.readlines():
        line = line.split("\t")
        fermi1_10.append([float(line[0]), float(line[1])])
fermi1_10 = np.array(fermi1_10)




'''
#################################################
Fermi integral
#################################################
'''

# General notation of Fermi integrals
# def fdint_robust(s: int, eta: float, T: float) -> float:
#     if eta/T > 709 and s == 1:
#         return eta**2/(2*T**2)
#     if s == 0:
#         return (float(np.log(1+np.exp(eta/T))))
#     if s == 0.5:
#         if (eta/T) < -10:
#             return float(0.886226925452758*1/(np.exp(-eta/T)+0.27))
#         elif (eta/T) > 5:
#             #return float (4/(3*np.sqrt(np.pi))*((eta/T)**2+((np.pi**2)/6))**0.75) # approximation with explicit formula
#             return float(0.886226925452758*(0.7522527780636751*((eta/T)**2+1.6449340668482264)**0.75)) # approximation with constant values pre-calculated
#         else:
#             pass
#     result = -mpmath.gamma(s+1)*mpmath.fp.polylog(s+1, -np.exp(eta/T))
#     return float(result.real) # important: it is not float-type before but: <class 'mpmath.ctx_mp_python.mpf'>
# fdint_vec = np.vectorize(fdint_robust)

# def fdint_robust(s: int, eta: float, T: float) -> float:
#     etaT = eta/T
#     if etaT > 709 and s == 1:
#         return eta**2/(2*T**2)
#     if s == 0:
#         return (float(np.log(1+np.exp(eta/T))))
#     if s == 0.5:
#         if (etaT) < -10:
#             return float(0.886226925452758*1/(np.exp(-etaT)+0.27))
#         if -10 <(etaT) < 0:
#             return float((0.67809872 + etaT * 0.2325289 + 0.031469379 * etaT**2 + 0.001983171 * etaT**3 + 4.8835524e-05 * etaT**4)/(1 + etaT *-0.44775237 + 0.151532 * etaT**2 + -0.032485253 * etaT**3 + 0.0033650243 * etaT**4 - -0.0003349541 *etaT**5))
#         if (etaT) > 9:
#             #return float (4/(3*np.sqrt(np.pi))*((eta/T)**2+((np.pi**2)/6))**0.75) # approximation with explicit formula
#             return float(0.886226925452758*(0.7522527780636751*((etaT)**2+1.6449340668482264)**0.75)) # approximation with constant values pre-calculated
#         if 0 <(etaT) < 9:
#             # return float(0.7221811 + etaT * 0.59851416 + 0.14596708 * etaT**2 + 0.0078053442 * etaT**3 + -0.00079834703 * etaT**4 + -3.2181832e-05 *etaT**5 + 3.388691e-06 * etaT**6)
#             return float((0.6811121 + etaT *0.35196675 + 0.071485242 * etaT**2 + 0.0064371623 * etaT**3 + 0.00021074676 * etaT**4)/(1 + etaT * -0.26460888 + 0.070236986 * etaT**2 + -0.0087561313 * etaT**3 + 0.00059522919 * etaT**4 - 1.642407e-05 *etaT**5))
#     result = -mpmath.gamma(s+1)*mpmath.fp.polylog(s+1, -np.exp(etaT))
#     return float(result.real) # important: it is not float-type before but: <class 'mpmath.ctx_mp_python.mpf'>
# fdint_vec = np.vectorize(fdint_robust)

def fdint_robust(s: int, eta: float, T: float) -> float:
    etaT = eta/T
    if etaT > 709 and s == 1:
        return eta**2/(2*T**2)
    if s == 0:
        return (float(np.log(1+np.exp(eta/T))))
    if s == 0.5:
        if (etaT) < -9.9:
            return float(0.886226925452758*1/(np.exp(-etaT)+0.27))
        if (etaT) > 19.9:
            return float(0.886226925452758*(0.7522527780636751*((etaT)**2+1.6449340668482264)**0.75)) # approximation with constant values pre-calculated
        else:
            # print(etaT,fermi1_10[int(etaT*1000+10000)][0])
            return fermi1_10[int(etaT*1000+10000)][1] + (fermi1_10[int(etaT*1000+10000+1)][1]-fermi1_10[int(etaT*1000+10000)][1]
                    )/(fermi1_10[int(etaT*1000+10000+1)][0]-fermi1_10[int(etaT*1000+10000)][0])*(etaT-fermi1_10[int(etaT*1000+10000)][0]) # approximation with constant values pre-calculated

    result = -mpmath.gamma(s+1)*mpmath.fp.polylog(s+1, -np.exp(etaT))
    return float(result.real) # important: it is not float-type before but: <class 'mpmath.ctx_mp_python.mpf'>
fdint_vec = np.vectorize(fdint_robust)


def fdint_robust2(s: int, eta: float, T: float) -> float:
    etaT = eta/T
    if s == 0:
        return (float(np.log(1+np.exp(etaT))))

    result = -mpmath.gamma(s+1)*mpmath.fp.polylog(s+1, -np.exp(etaT))
    # result = -mpmath.fp.polylog(s+1, -np.exp(eta/T))
    return float(result.real) # important: it is not float-type before but: <class 'mpmath.ctx_mp_python.mpf'>
fdint_vec2 = np.vectorize(fdint_robust2)

# Calculates Fermi integrals for a single band
def spb_fermiInt_vs_temp(s: int, T: np.ndarray, mass_sign: int, fermi_energy: float) -> list[float]:
    last_mu = fermi_energy
    fermiInt = np.zeros(shape=len(T))
    
    for i, temp in enumerate(T): 
        mu = fsolve(spb_chemPot, [last_mu], (temp, mass_sign, fermi_energy))[0]
        last_mu = mu
        
        fermiInt[i] = fdint_vec(s + lamb, mass_sign * mu, temp)
        
    return fermiInt

# Calculates Fermi integrals for two bands
def dpb_fermiInt_vs_temp(s: float, T: float, mass: float, gap: float, fermi_energy: float, degen: int) -> tuple[list[float]]:
    last_mu = fermi_energy
    fermiInt_1 = np.zeros(shape=len(T))
    fermiInt_2 = np.zeros(shape=len(T))
    
    for i, temp in enumerate(T):       
        mu = fsolve(dpb_chemPot, [last_mu], (temp,gap,mass,fermi_energy, degen))[0]
        last_mu = mu

        fermiInt_1[i] = fdint_vec(s + lamb, -mu, temp)
        fermiInt_2[i] = fdint_vec(s + lamb, np.sign(mass) * (mu - gap), temp)
        
    return fermiInt_1, fermiInt_2


'''
#################################################
Basic thermoelectric functions
#################################################
'''

# General formula for calculating the Onsager coefficient of a single band
def onsi_general(eta: float, meff: float, K: float, beta: float, T: float):
    result = - np.sign(meff) * (kB/e) * K * T**(beta + 1) / abs(meff) * ((lamb + 2) / (lamb + 1) * fdint_vec(1+lamb,np.sign(meff)*eta,T) - np.sign(meff) * fdint_vec(0+lamb,np.sign(meff)*eta,T) * eta/T)
    return result

# General formula for calculating the electrical conductivity of a single band
def sigma_general(eta: float, meff: float, K: float, beta: float, T: float) -> float:
    #Temperature factor defined as beta + 1, since 1 is obtained from partial integration instead of 3/2
    result = K * T**(beta + 1) / abs(meff) * fdint_vec(0+lamb,np.sign(meff)*eta,T) 
    return result


'''
#################################################
Useful functions
#################################################
'''

# Calculates the charge carrier concentration of a single band
def spb_carCon(eta: float, meff: float, T: float) -> float:
    if T==0:
        # If Fermi energy is inside the band ...
        if eta*np.sign(meff) > 0:
            return (4*np.pi*(2*np.abs(meff)*me*kB*np.abs(eta)/h**2)**1.5)*(2/3) * 1e-6
            #return ((2*np.abs(meff)*me*kB*T)**1.5/(2*np.pi**2*hbar**3)**1.5)*(2/3) * 1e-6
    
        # If Fermi energy is outside the band ...
        else:
            return 0     
    else:  
        return 4*np.pi*(2*np.abs(meff)*me*kB*T/h**2)**1.5 * fdint_vec(0.5,np.sign(meff)*eta,T) * 1e-6
        #return (2*(np.abs(meff)*me*kB*T)**1.5/(2*np.pi**2*hbar**3)) * fdint_vec(0.5,np.sign(meff)*eta,T) * 1e-6
    
# Calculates the chemical potential of a single band at a temperature T
def spb_chemPot(eta_here: list[float], temp: float, band_mass: int, fermi_energy: float) -> float:
    result = spb_carCon(eta_here[0], band_mass,temp) - spb_carCon(fermi_energy, band_mass,0)
    return np.abs(result)

# Calculates the temperature-dependent chemical potential of a single band
def spb_chemPot_vs_temp(temp_array: np.ndarray, band_mass: int, fermi_energy: float) -> float:
    chemPot_array = np.zeros(shape=len(temp_array))
    last_mu = fermi_energy
    
    for i, temp in enumerate(temp_array):
        mu = fsolve(spb_chemPot, [last_mu], (temp, band_mass, fermi_energy))[0]
        last_mu = mu
        chemPot_array[i] = mu
        
    return chemPot_array

# Calculates the chemical potential of two bands at a temperature T with variable degeneracy ratio degen
def dpb_chemPot(eta_here: list[float], temp: float, gap: float, meff: float, fermi_energy: float, degen: float) -> float:
    result = (-1)*spb_carCon(eta_here[0],-1,temp) + np.sign(meff)*spb_carCon(eta_here[0]-gap,meff*degen**(5/3),temp) - ((-1)*spb_carCon(fermi_energy,-1,0) + np.sign(meff)*spb_carCon(fermi_energy-gap,meff*degen**(5/3),0))
    return np.abs(result)

# Calculates the temperature-dependent chemical potential of two bands
def dpb_chemPot_vs_temp(temp_array: np.ndarray, gap: float, meff: float, fermi_energy: float, degen: float) -> float:
    chemPot_array = np.zeros(shape=len(temp_array))
    last_mu = fermi_energy

    for i, temp in enumerate(temp_array):
        mu = fsolve(dpb_chemPot, [last_mu], (temp, gap, meff, fermi_energy, degen))[0]
        last_mu = mu
        chemPot_array[i] = mu
        
    return chemPot_array

# Calculates the temperature-dependent Lorenz number of a single band
def spb_lorNum_vs_temp(temp_array: np.ndarray, fermi_energy: float, band_mass: float) -> np.ndarray:
    fermiInt_0 = spb_fermiInt_vs_temp(0, temp_array, np.sign(band_mass), fermi_energy)
    fermiInt_1 = spb_fermiInt_vs_temp(1, temp_array, np.sign(band_mass), fermi_energy)
    fermiInt_2 = spb_fermiInt_vs_temp(2, temp_array, np.sign(band_mass), fermi_energy)

    return (kB/e)**2 * ((1 + lamb) * (3 + lamb) * fermiInt_0 * fermiInt_2 - (2 + lamb)**2 * fermiInt_1**2) / ((1 + lamb)**2 * fermiInt_0**2)

# Calculates the individual temperature-dependent Lorenz numbers of two bands
def dpb_lorNum_vs_temp(temp_array: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float) -> tuple[np.ndarray]:
    f0_1, f0_2 = dpb_fermiInt_vs_temp(0, temp_array, meff, gap, fermi_energy, degen)
    f1_1, f1_2 = dpb_fermiInt_vs_temp(1, temp_array, meff, gap, fermi_energy, degen)
    f2_1, f2_2 = dpb_fermiInt_vs_temp(2, temp_array, meff, gap, fermi_energy, degen)

    lorNum_1 = (kB/e)**2 * ((1 + lamb) * (3 + lamb) * f0_1 * f2_1 - (2 + lamb)**2 * f1_1**2) / ((1 + lamb)**2 * f0_1**2)
    lorNum_2 = (kB/e)**2 * ((1 + lamb) * (3 + lamb) * f0_2 * f2_2 - (2 + lamb)**2 * f1_2**2) / ((1 + lamb)**2 * f0_2**2)

    return lorNum_1, lorNum_2


'''
#################################################
Functions used for calculating the thermoelectric properties
#################################################
'''

'''
SPB
'''

# Calculates the Seebeck coefficient of a single band
def spb_see_vs_temp(temp_array: np.ndarray, fermi_energy: float, band_mass: int, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the Seebeck coefficient of a single bands at various temperatures for the only-Seebeck mode
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    fermi_energy : float
        Fitting parameter, representing the position of the Fermi energy with respect to the maximum of the band
    band_mass : int
        Mass of the band
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Seebeck coefficient values
    """
    
    last_mu = fermi_energy
    sig = np.zeros(shape=len(temp_array))
    ons = np.zeros(shape=len(temp_array))   
    see = np.zeros(shape=len(temp_array))
    
    for i, temp in enumerate(temp_array):
        
        mu = fsolve(spb_chemPot, [last_mu], (temp, band_mass, fermi_energy))[0]
        last_mu = mu
        
        sig[i] = sigma_general(eta = mu, meff = band_mass, K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons[i] = onsi_general(eta = mu,  meff = band_mass, K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    see = scal_fac * ons / sig
        
    return see

# Calculates the electrical resistivity of a single band for acoustic-phonon scattering
def spb_rho_vs_temp_acPh(temp_array: np.ndarray, sig: np.ndarray, para_A: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the resistivity of a single band at various tempeatures for a given acoustic-phonon scattering parameter.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Fitting parameter, representing the prefactor of acoustic-phonon scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of resistivity values
    """

    return scal_fac / ( para_A * sig )

# Calculates the electrical conductivity of a single band for acoustic-phonon scattering
def spb_elecCond_vs_temp_acPh(temp_array: np.ndarray, sig: np.ndarray, para_A: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electrical conductivity of a single band at various temperatures for a given acoustic-phonon scattering parameter.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Fitting parameter, representing the prefactor of acoustic-phonon scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electrical conductivity values
    """

    return scal_fac * para_A * sig

# Calculates the electron thermal conductivity of a single band for acoustic-phonon scattering
def spb_thermCond_vs_temp_acPh(temp_array: np.ndarray, sig: np.ndarray, fermi_energy: float, band_mass: float, para_A: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electron thermal conductivity of a single band at various temperatures for a given acoustic-phonon scattering parameter.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Fitting parameter, representing the prefactor of acoustic-phonon scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electron thermal conductivity values
    """

    return spb_elecCond_vs_temp_acPh(temp_array, sig, para_A, scal_fac) * spb_lorNum_vs_temp(temp_array, fermi_energy, band_mass) * temp_array

# Calculates the electrical resistivity of a single band for alloy-disorder scattering
def spb_rho_vs_temp_dis(temp_array: np.ndarray, sig: np.ndarray, para_A: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the resistivity of a single band at various temperatures for a given alloy-disorder scattering parameter.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Fitting parameter, representing the prefactor of alloy-disorder scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of resistivity values
    """

    return scal_fac / ( para_A * temp_array * sig )

# Calculates the electrical conductivity of a single band for alloy-disorder scattering
def spb_elecCond_vs_temp_dis(temp_array: np.ndarray, sig: np.ndarray, para_A: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electrical conductivity of a single band at various temperatures for a given alloy-disorder scattering parameter.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Fitting parameter, representing the prefactor of alloy-disorder scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electrical conductivity values
    """

    return scal_fac * para_A * temp_array * sig

# Calculates the electron thermal conductivity of a single band for alloy-disorder scattering
def spb_thermCond_vs_temp_dis(temp_array: np.ndarray, sig: np.ndarray, fermi_energy: float, band_mass: float, para_A: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electron thermal conductivity of a single band at various temperatures for a given alloy-disorder scattering parameter.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Fitting parameter, representing the prefactor of alloy-disorder scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electron thermal conductivity values
    """

    return spb_elecCond_vs_temp_dis(temp_array, sig, para_A, scal_fac) * spb_lorNum_vs_temp(temp_array, fermi_energy, band_mass) * temp_array

# Calculates the electrical resistivity of a single band for acoustic-phonon and alloy-disorder scattering
def spb_rho_vs_temp_acPh_dis(T: np.ndarray, sig: np.ndarray, para_A: float, para_B: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the resistivity of a single band at various temperatures for a given set of acoustic-phonon and alloy-disorder scattering parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by T
    para_A : float
        First fitting parameter, representing the prefactor of acoustic-phonon scattering
    para_B : float
        Second fitting parameter, representing the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of resistivity values
    """

    return scal_fac / (para_A * T / (para_B + T) * sig)

# Calculates the electrical conductivity of a single band for acoustic-phonon and alloy-disorder scattering
def spb_elecCond_vs_temp_acPh_dis(temp_array: np.ndarray, sig: np.ndarray, para_A: float, para_B: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electrical conductivity of a single band at various temperatures for a given set of acoustic-phonon and alloy-disorder scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        First fitting parameter, representing the prefactor of acoustic-phonon scattering
    para_B : float
        Second fitting parameter, representing the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electrical conductivity values
    """

    return scal_fac * para_A * temp_array / (para_B + temp_array) * sig

# Calculates the electron thermal conductivity of a single band for acoustic-phonon and alloy-disorder scattering
def spb_thermCond_vs_temp_acPh_dis(temp_array: np.ndarray, sig: np.ndarray, fermi_energy: float, band_mass: float, para_A: float, para_B: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electron thermal conductivity of a single band at various temperatures for a given set of acoustic-phonon and alloy-disorder scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig : numpy.ndarray
        'Pristine' conductivity of the band, calculated with mass = mass_sign, K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        First fitting parameter, representing the prefactor of acoustic-phonon scattering
    para_B : float
        Second fitting parameter, representing the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electron thermal conductivity values
    """

    return spb_elecCond_vs_temp_acPh_dis(temp_array, sig, para_A, para_B, scal_fac) * spb_lorNum_vs_temp(temp_array, fermi_energy, band_mass) * temp_array

# Calculates the Hall coefficient of a single band
def spb_hall_vs_temp(T: np.ndarray, fermi_energy: float, mass_sign: int, para_E: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the Hall coefficient of a single band at various temperatures
    
    Parameters
    ----------
    T : numpy.ndarray
        Numpy array containing all temperatures
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of the band
    mass_sign : int
        Sign of the band mass; -1 vor valence band, +1 for conduction band
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Hall coefficient values
    """
    
    factor = 3 * hbar**3 * np.pi**2 / (e * (2 * kB * me)**(3/2)) * (0.5 + 2 * lamb) / (1 + lamb)**2
    return factor * T**(-1.5) * scal_fac / para_E * fdint_vec(-0.5 + lamb, mass_sign * fermi_energy, T) / fdint_vec(0 + lamb, mass_sign * fermi_energy, T)**2

'''
DPB
'''

# Calculates the Seebeck coefficient of two bands for the Seebeck-only mode
def dpb_see_vs_temp(T: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the Seebeck coefficient of two bands at various temperatures for the Seebeck-only mode
    
    Parameters
    ----------
    T : numpy.ndarray
        Numpy array containing all temperatures
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the inverse degeneracies
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Seebeck coefficient values
    """
    
    last_mu = fermi_energy
    sig1 = np.zeros(shape=len(T))
    sig2 = np.zeros(shape=len(T))
    ons1 = np.zeros(shape=len(T))
    ons2 = np.zeros(shape=len(T))    
    see = np.zeros(shape=len(T))
    
    for i, temp in enumerate(T):
        
        mu = fsolve(dpb_chemPot, [last_mu], (temp, gap, meff, fermi_energy, degen))[0]
        last_mu = mu
        
        factor = 1 / meff
        
        sig1[i] = sigma_general(eta = mu,       meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        sig2[i] = sigma_general(eta = mu - gap, meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
        ons1[i] = onsi_general(eta = mu,        meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons2[i] = onsi_general(eta = mu - gap,  meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    see = scal_fac * (ons1 + factor * ons2)/(sig1 + factor * sig2)
        
    return see

# Calculates the Seebeck coefficient of two bands if scattering times of acoustic-phonon and alloy-disorder scattering are considered
def dpb_see_vs_temp_double_scatter(temp_array: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float, para_B: float, para_C: float, para_D: float, scal_fac: float = 1.0) -> np.ndarray: #args is supposed to look like: mass1,mass2,....,gap1,gap2,..Nv1, Nv2...,fermienergy):# *args is a tuple or arguments, i.e. a variable number of arguments
    """Calculates the Seebeck coefficient of two bands at various temperatures for a given set of band-structure and scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    meff : float
        First fitting parameter, representing the ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    gap : numpy.ndarray
        Second fitting parameter, representing the band gap, i.e. the distance between the minima/maxima of the parabolas
    ef : float
        Third fitting parameter, representing the position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    para_B : float
        Parameter fitted in 'rho_vs_temp_acPh_dis', representing the ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    para_C : float
        Parameter fitted in 'rho_vs_temp_acPh_dis', representing the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 1
    para_D : float
        Parameter fitted in 'rho_vs_temp_acPh_dis', representing the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 2
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Seebeck coefficient values
        
    See also
    -------
    rho_vs_temp_acPh_dis : Returns an array of resistivity values
    """

    sig1 = np.zeros(shape=len(temp_array))
    sig2 = np.zeros(shape=len(temp_array))
    ons1 = np.zeros(shape=len(temp_array))
    ons2 = np.zeros(shape=len(temp_array))    
    see = np.zeros(shape=len(temp_array))
    
    chemPot_array = dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)
    factor = para_B * ( para_C + temp_array ) / ( para_D + temp_array ) / meff
    
    for i, temp in enumerate(temp_array):

        factor = para_B * ( para_C + temp ) / ( para_D + temp ) / meff
        
        sig1[i] = sigma_general(eta = chemPot_array[i],       meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        sig2[i] = sigma_general(eta = chemPot_array[i] - gap, meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
        ons1[i] = onsi_general(eta = chemPot_array[i],        meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons2[i] = onsi_general(eta = chemPot_array[i] - gap,  meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    see = scal_fac * (ons1 + factor * ons2)/(sig1 + factor * sig2)
        
    return see

# Calculates the Seebeck coefficient of two bands if scattering times of either acoustic-phonon or alloy-disorder scattering are considered
def dpb_see_vs_temp_single_scatter(temp_array: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float, para_B: float, scal_fac: float = 1.0) -> np.ndarray: #args is supposed to look like: mass1,mass2,....,gap1,gap2,..Nv1, Nv2...,fermienergy):# *args is a tuple or arguments, i.e. a variable number of arguments
    """Calculates the Seebeck coefficient of two bands at various temperatures for a given set of band-structure scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    meff : float
        First fitting parameter, representing the ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    gap : numpy.ndarray
        Second fitting parameter, representing the band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Third fitting parameter, representing the position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    para_B : float
        Parameter fitted in 'dpb_rho_vs_temp_acPh' or 'dpb_rho_vs_temp_dis', representing the ratio of the prefactors of either acoustic-phonon or alloy-disorder scattering of band 2 to band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Seebeck coefficient values
    """
    
    sig1 = np.zeros(shape=len(temp_array))
    sig2 = np.zeros(shape=len(temp_array))
    ons1 = np.zeros(shape=len(temp_array))
    ons2 = np.zeros(shape=len(temp_array))    
    see = np.zeros(shape=len(temp_array))
    
    chemPot_array = dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)
    factor = para_B / meff
    
    for i, temp in enumerate(temp_array):

        sig1[i] = sigma_general(eta = chemPot_array[i],       meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        sig2[i] = sigma_general(eta = chemPot_array[i] - gap, meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
        ons1[i] = onsi_general(eta = chemPot_array[i],        meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons2[i] = onsi_general(eta = chemPot_array[i] - gap,  meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    see = scal_fac * (ons1 + factor * ons2)/(sig1 + factor * sig2)
        
    return see

# Calculates the electrical resistivity of two bands for acoustic-phonon scattering
def dpb_rho_vs_temp_acPh(T: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, meff: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the resistivity of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by T
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by T
    para_A : float
        First fitting parameter, representing the prefactor of acoustic-phonon scattering of band 1
    para_B : float
        Second fitting parameter, representing the ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    meff : float
        Parameter, representing the ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of resistivity values
    """

    return scal_fac / ( para_A * ( f0_1 + 1 / meff * para_B * f0_2 ) )

# Calculates the electrical resistivity of two bands for acoustic-phonon scattering
def dpb_rho_vs_temp_acPh_new(T: np.ndarray, para_A: float, meff: float, initial_weight: float, gap: float, fermi_energy: float, degen: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the resistivity of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Numpy array containing all temperatures
    para_A : float
        First fitting parameter, representing the prefactor of acoustic-phonon scattering of band 1
    meff : float
        Parameter, representing the ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    XXXX
    YYYY
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of resistivity values
    """
    
    f0_1, f0_2 = dpb_fermiInt_vs_temp(s = 0, T = T, mass = meff, gap = gap, fermi_energy = fermi_energy, degen = degen)

    return scal_fac / ( para_A * ( f0_1 + 1 / initial_weight * f0_2 ) )

# Calculates the electrical resistivity of two bands for alloy-disorder scattering
def dpb_rho_vs_temp_dis(T: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, meff: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the resistivity of two bands at various temperatures for a given set of alloy-disorder scattering parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by T
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by T
    para_A : float
        First fitting parameter, representing the prefactor of alloy-disorder scattering of band 1
    para_B : float
        Second fitting parameter, representing the ratio of the prefactors of alloy-disorder scattering of band 2 to band 1
    meff : float
        Parameter, representing the ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of resistivity values
    """

    return scal_fac / ( para_A * T * ( f0_1 + 1 / meff * para_B * f0_2 ) )

# Calculates the electrical resistivity of two bands for acoustic-phonon and alloy-disorder scattering
def dpb_rho_vs_temp_acPh_dis(T: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, para_C: float, para_D: float, meff: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the resistivity of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by T
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by T
    para_A : float
        First fitting parameter, representing the prefactor of acoustic-phonon scattering of band 1
    para_B : float
        Second fitting parameter, representing the ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    para_C : float
        Third fitting parameter, representing the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 1
    para_D : float
        Fourth fitting parameter, representing the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 2
    meff : float
        Parameter, representing the ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of resistivity values
    """

    return scal_fac / ( para_A * T * ( f0_1 / (para_C + T) + 1 / meff * para_B * f0_2 / (para_D + T) ) )

# Calculates the electron thermal conductivity of two bands for acoustic-phonon scattering
def dpb_thermCond_vs_temp_acPh(temp_array: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, meff: float, gap: float, fermi_energy: float, degen: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electron thermal conductivity of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by temp_array
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Prefactor of alloy-disorder scattering of band 1
    para_B : float
        Ratio of the prefactors of alloy-disorder scattering of band 2 to band 1
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the inverse degeneracies
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electron thermal conductivity values
    """

    sig1, sig2 = dpb_ind_elecCond_vs_temp_acPh(temp_array, f0_1, f0_2, para_A, para_B, meff, scal_fac = 1.0)
    see1, see2 = dpb_ind_see_vs_temp(temp_array, meff, gap, fermi_energy, degen, scal_fac = 1.0)
    lorNum_1, lorNum_2 = dpb_lorNum_vs_temp(temp_array, meff, gap, fermi_energy, degen)

    return scal_fac * temp_array * (lorNum_1 * sig1 + lorNum_2 * sig2) + temp_array * (sig1 * see1**2 + sig2 * see2**2 - (sig1 * see1 + sig2 * see2)**2 / (sig1 + sig2))

# Calculates the electron thermal conductivity of two bands for alloy-disorder scattering
def dpb_thermCond_vs_temp_dis(temp_array: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, meff: float, gap: float, fermi_energy: float, degen: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electron thermal conductivity of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by temp_array
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Prefactor of alloy-disorder scattering of band 1
    para_B : float
        Ratio of the prefactors of alloy-disorder scattering of band 2 to band 1
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the inverse degeneracies
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electron thermal conductivity values
    """

    sig1, sig2 = dpb_ind_elecCond_vs_temp_dis(temp_array, f0_1, f0_2, para_A, para_B, meff, scal_fac = 1.0)
    see1, see2 = dpb_ind_see_vs_temp(temp_array, meff, gap, fermi_energy, degen, scal_fac = 1.0)
    lorNum_1, lorNum_2 = dpb_lorNum_vs_temp(temp_array, meff, gap, fermi_energy, degen)

    return scal_fac * temp_array * (lorNum_1 * sig1 + lorNum_2 * sig2) + temp_array * (sig1 * see1**2 + sig2 * see2**2 - (sig1 * see1 + sig2 * see2)**2 / (sig1 + sig2))

# Calculates the electron thermal conductivity of two bands for acoustic-phonon and alloy-disorder scattering
def dpb_thermCond_vs_temp_acPh_dis(temp_array: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, para_C: float, para_D: float, meff: float, gap: float, fermi_energy: float, degen: float, scal_fac: float = 1.0) -> np.ndarray:
    """Calculates the electron thermal conductivity of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by temp_array
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Prefactor of alloy-disorder scattering of band 1
    para_B : float
        Ratio of the prefactors of alloy-disorder scattering of band 2 to band 1
    para_C : float
        Ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 1
    para_D : float
        Ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 2
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the inverse degeneracies
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied
        
    Returns
    -------
    out : numpy.ndarray
        Numpy array of electron thermal conductivity values
    """

    sig1, sig2 = dpb_ind_elecCond_vs_temp_acPh_dis(temp_array, f0_1, f0_2, para_A, para_B, para_C, para_D, meff, scal_fac = 1.0)
    see1, see2 = dpb_ind_see_vs_temp(temp_array, meff, gap, fermi_energy, degen, scal_fac = 1.0)
    lorNum_1, lorNum_2 = dpb_lorNum_vs_temp(temp_array, meff, gap, fermi_energy, degen)

    return scal_fac * temp_array * (lorNum_1 * sig1 + lorNum_2 * sig2) + temp_array * (sig1 * see1**2 + sig2 * see2**2 - (sig1 * see1 + sig2 * see2)**2 / (sig1 + sig2))

# Calculates the Hall coefficient of two bands for acoustic-phonon scattering
def dpb_hall_vs_temp_acPh(T: np.ndarray, rho: np.ndarray, f_m05_1: np.ndarray, f_m05_2: np.ndarray, meff: float, degen: float, para_A: float, para_B: float, para_E: float, scal_fac: float = 1.0) -> np.ndarray:
    """
    Calculates the Hall coefficient of two bands at various temperatures for a given set of band-structure, scattering and mass parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Temperature
    rho : numpy.ndarray
        Total resistivity at the temperatures given by T
    f_m05_1 : numpy.ndarray
        Fermi integral of -0.5th order of band 1 at the temperatures given by T
    f_m05_2 : numpy.ndarray
        Fermi integral of -0.5th order of band 2 at the temperatures given by T
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    para_A : float
        Scattering-related parameter, proportional to the prefactor of acoustic-phonon scattering of band 1
    para_B : float
        Scattering-related parameter, proportional to the ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    para_E : float
        Fitting parameter, proportional to the mass of band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied for improved fitting convergence
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Hall coefficient values
    """

    factor = 3 * hbar**3 * np.pi**2 / (2 * e * (2 * kB * me)**(3/2)) * (0.5 + 2 * lamb) / (1 + lamb)**2
    return factor * T**(-1.5) * scal_fac * rho**2 * para_A**2 / para_E * (f_m05_1 - np.sign(meff) / degen**(5/2) / (np.abs(meff))**(3/2) * (para_B / meff)**2 * f_m05_2) 

# Calculates the Hall coefficient of two bands for alloy-disorder scattering
def dpb_hall_vs_temp_dis(T: np.ndarray, rho: np.ndarray, f_m05_1: np.ndarray, f_m05_2: np.ndarray, meff: float, degen: float, para_A: float, para_B: float, para_E: float, scal_fac: float = 1.0) -> np.ndarray:
    """
    Calculates the Hall coefficient of two bands at various temperatures for a given set of band-structure, scattering and mass parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Temperature
    rho : numpy.ndarray
        Total resistivity at the temperatures given by T
    f_m05_1 : numpy.ndarray
        Fermi integral of -0.5th order of band 1 at the temperatures given by T
    f_m05_2 : numpy.ndarray
        Fermi integral of -0.5th order of band 2 at the temperatures given by T
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    para_A : float
        Scattering-related parameter, proportional to the prefactor of acoustic-phonon scattering of band 1
    para_B : float
        Scattering-related parameter, proportional to the ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    para_E : float
        Free parameter, proportional to the mass of band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied for improved fitting convergence
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Hall coefficient values
    """

    factor = 3 * hbar**3 * np.pi**2 / (2 * e * (2 * kB * me)**(3/2)) * (0.5 + 2 * lamb) / (1 + lamb)**2
    return factor * T**(0.5) * scal_fac * rho**2 * para_A**2 / para_E * (f_m05_1 - np.sign(meff) / degen**(5/2) / (np.abs(meff))**(3/2) * (para_B / meff)**2 * f_m05_2)

# Calculates the Hall coefficient of two bands for acoustic-phonon and alloy-disorder scattering
def dpb_hall_vs_temp_acPh_dis(T: np.ndarray, rho: np.ndarray, f_m05_1: np.ndarray, f_m05_2: np.ndarray, meff: float, degen: float, para_A: float, para_B: float, para_C: float, para_D: float, para_E: float, scal_fac: float = 1.0) -> np.ndarray:
    """
    Calculates the Hall coefficient of two bands at various temperatures for a given set of band-structure, scattering and mass parameters.
    
    Parameters
    ----------
    T : numpy.ndarray
        Temperature
    rho : numpy.ndarray
        Total resistivity at the temperatures given by T
    f_m05_1 : numpy.ndarray
        Fermi integral of -0.5th order of band 1 at the temperatures given by T
    f_m05_2 : numpy.ndarray
        Fermi integral of -0.5th order of band 2 at the temperatures given by T
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    para_A : float
        Scattering-related parameter, proportional to the prefactor of acoustic-phonon scattering of band 1
    para_B : float
        Scattering-related parameter, proportional to the ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    para_C : float
        Scattering-related parameter, proportional to the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 1
    para_D : float
        Scattering-related parameter, proportional to the ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 2
    para_E : float
        Fitting parameter, proportional to the mass of band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied for improved fitting convergence
    Returns
    -------
    out : numpy.ndarray
        Numpy array of Hall coefficient values
    """

    factor = 3 * hbar**3 * np.pi**2 / (2 * e * (2 * kB * me)**(3/2)) * (0.5 + 2 * lamb) / (1 + lamb)**2
    return factor * T**(0.5) * scal_fac * rho**2 * para_A**2 / para_E * ((1/(para_C + T))**2 * f_m05_1 - np.sign(meff) / degen**(5/2) / (np.abs(meff))**(3/2) * (para_B / meff / (para_D + T))**2 * f_m05_2)

'''
Individual contributions
'''

# Calculates the individual Seebeck coefficients of two bands
def dpb_ind_see_vs_temp(temp_array: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float, scal_fac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the individual Seebeck coefficients of two bands at various temperatures for the Seebeck-only mode
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the inverse degeneracies
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    scal_fac : float
        Scaling factor, by which both functions are multiplied
        
    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        Tuple of numpy array of individual Seebeck coefficients
    """
    
    sig1 = np.zeros(shape=len(temp_array))
    sig2 = np.zeros(shape=len(temp_array))
    ons1 = np.zeros(shape=len(temp_array))
    ons2 = np.zeros(shape=len(temp_array))    
    
    chemPot_array = dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)

    for i, temp in enumerate(temp_array):
        
        sig1[i] = sigma_general(eta = chemPot_array[i],       meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        sig2[i] = sigma_general(eta = chemPot_array[i] - gap, meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
        ons1[i] = onsi_general(eta = chemPot_array[i],        meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons2[i] = onsi_general(eta = chemPot_array[i] - gap,  meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    ind_see1 = ons1 / sig1
    ind_see2 = ons2 / sig2
        
    return ind_see1, ind_see2

# Calculates the individual contributions to the Seebeck coefficient of two bands for the Seebeck-only mode
def dpb_indCont_see_vs_temp(temp_array: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float, scal_fac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the individual contributions to the Seebeck coefficient of two bands at various temperatures for the Seebeck-only mode
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the inverse degeneracies
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    scal_fac : float
        Scaling factor, by which both functions are multiplied
        
    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        Tuple of numpy array of individual Seebeck coefficient contributions
    """
    
    sig1 = np.zeros(shape=len(temp_array))
    sig2 = np.zeros(shape=len(temp_array))
    ons1 = np.zeros(shape=len(temp_array))
    ons2 = np.zeros(shape=len(temp_array))    
    
    chemPot_array = dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)
    factor = 1 / meff
    
    for i, temp in enumerate(temp_array):
        
        sig1[i] = sigma_general(eta = chemPot_array[i],       meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        sig2[i] = sigma_general(eta = chemPot_array[i] - gap, meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
        ons1[i] = onsi_general(eta = chemPot_array[i],        meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons2[i] = onsi_general(eta = chemPot_array[i] - gap,  meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    ind_see1 = scal_fac * ons1 / (sig1 + factor * sig2)
    ind_see2 = scal_fac * factor * ons2 / (sig1 + factor * sig2)
        
    return ind_see1, ind_see2

# Calculates the individual contributions to the Seebeck coefficient of two bands if scattering times of acoustic-phonon and alloy-disorder scattering are considered
def dpb_indCont_see_vs_temp_double_scatter(temp_array: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float, para_B: float, para_C: float, para_D: float, scal_fac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the individual contributions to the Seebeck coefficient of two bands at various temperatures for a given set of band-structure and scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    para_B : float
        Ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    para_C : float
        Ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 1
    para_D : float
        Ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 2
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        Tuple of numpy array of individual Seebeck coefficient contributions
    """
    
    sig1 = np.zeros(shape=len(temp_array))
    sig2 = np.zeros(shape=len(temp_array))
    ons1 = np.zeros(shape=len(temp_array))
    ons2 = np.zeros(shape=len(temp_array))
    
    chemPot_array = dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)
    factor = para_B * ( para_C + temp_array ) / ( para_D + temp_array ) / meff
    
    for i, temp in enumerate(temp_array):
        
        sig1[i] = sigma_general(eta = chemPot_array[i],       meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        sig2[i] = sigma_general(eta = chemPot_array[i] - gap, meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
        ons1[i] = onsi_general(eta = chemPot_array[i],        meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons2[i] = onsi_general(eta = chemPot_array[i] - gap,  meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    ind_see1 = scal_fac * ons1 / (sig1 + factor * sig2)
    ind_see2 = scal_fac * factor * ons2 / (sig1 + factor * sig2)
        
    return ind_see1, ind_see2

# Calculates the individual contributions to the Seebeck coefficient of two bands if scattering times of either acoustic-phonon or alloy-disorder scattering are considered
def dpb_indCont_see_vs_temp_single_scatter(temp_array: np.ndarray, meff: float, gap: float, fermi_energy: float, degen: float, para_B: float, scal_fac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the individual contributions to the Seebeck coefficient of two bands at various temperatures for a given set of band-structure scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    gap : numpy.ndarray
        Band gap, i.e. the distance between the minima/maxima of the parabolas
    fermi_energy : float
        Position of the Fermi energy with respect to the maximum of band 1
    degen : float
        Ratio of the degeneracy of band 2 to band 1
    para_B : float
        Parameter fitted in 'dpb_rho_vs_temp_acPh' or 'dpb_rho_vs_temp_dis', representing the ratio of the prefactors of either acoustic-phonon or alloy-disorder scattering of band 2 to band 1
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        Tuple of numpy array of individual Seebeck coefficient contributions
    """
    
    sig1 = np.zeros(shape=len(temp_array))
    sig2 = np.zeros(shape=len(temp_array))
    ons1 = np.zeros(shape=len(temp_array))
    ons2 = np.zeros(shape=len(temp_array))
    
    chemPot_array = dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)
    factor = para_B / meff
    
    for i, temp in enumerate(temp_array):

        sig1[i] = sigma_general(eta = chemPot_array[i],       meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        sig2[i] = sigma_general(eta = chemPot_array[i] - gap, meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
        ons1[i] = onsi_general(eta = chemPot_array[i],        meff = -1,            K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        ons2[i] = onsi_general(eta = chemPot_array[i] - gap,  meff = np.sign(meff), K = 1, beta = -1, T = temp) #beta = -1 to remove any temperature dependence that would cancel out eventually
        
    ind_see1 = scal_fac * ons1 / (sig1 + factor * sig2)
    ind_see2 = scal_fac * factor * ons2 / (sig1 + factor * sig2)
        
    return ind_see1, ind_see2

# Calculates the individual electrical conductivities of two bands for acoustic-phonon scattering
def dpb_ind_elecCond_vs_temp_acPh(temp_array: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, meff: float, scal_fac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the individual electrical conductivities of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by temp_array
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Prefactor of acoustic-phonon scattering of band 1
    para_B : float
        Ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        Tuple of numpy arrays of individual electrical conductivities
    """

    sig1 = scal_fac * para_A * f0_1
    sig2 = scal_fac * para_A * 1 / meff * para_B * f0_2

    return sig1, sig2

# Calculates the individual electrical conductivities of two bands for alloy-disorder scattering
def dpb_ind_elecCond_vs_temp_dis(temp_array: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, meff: float, scal_fac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the individual electrical conductivities of two bands at various temperatures for a given set of alloy-disorder scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by temp_array
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Prefactor of alloy-disorder scattering of band 1
    para_B : float
        Ratio of the prefactors of alloy-disorder scattering of band 2 to band 1
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        Tuple of numpy arrays of individual electrical conductivities
    """

    sig1 = scal_fac * para_A * temp_array * f0_1
    sig2 = scal_fac * para_A * temp_array * 1 / meff * para_B * f0_2

    return sig1, sig2

# Calculates the individual electrical conductivities of two bands for acoustic-phonon and alloy-disorder scattering
def dpb_ind_elecCond_vs_temp_acPh_dis(temp_array: np.ndarray, f0_1: np.ndarray, f0_2: np.ndarray, para_A: float, para_B: float, para_C: float, para_D: float, meff: float, scal_fac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the individual electrical conductivities of two bands at various temperatures for a given set of scattering parameters.
    
    Parameters
    ----------
    temp_array : numpy.ndarray
        Numpy array containing all temperatures
    sig1 : numpy.ndarray
        'Pristine' conductivity of band 1, calculated with mass = -1, K = 1 and beta = 0 at the temperatures given by temp_array
    sig2 : numpy.ndarray
        'Pristine' conductivity of band 2, calculated with mass = numpy.sign(meff), K = 1 and beta = 0 at the temperatures given by temp_array
    para_A : float
        Prefactor of acoustic-phonon scattering of band 1
    para_B : float
        Ratio of the prefactors of acoustic-phonon scattering of band 2 to band 1
    para_C : float
        Ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 1
    para_D : float
        Ratio of the prefactors of acoustic-phonon scattering to alloy-disorder scattering of band 2
    meff : float
        Ratio of the band mass of band 2 to band 1, weighted by the degeneracy
    scal_fac : float
        Scaling factor, by which the function is multiplied for improvied fitting convergence
        
    Returns
    -------
    out : numpy.ndarray, numpy.ndarray
        Tuple of numpy arrays of individual electrical conductivities
    """

    sig1 = scal_fac * para_A * temp_array * f0_1 / (para_C + temp_array)
    sig2 = scal_fac * para_A * temp_array * 1 / meff * para_B * f0_2 / (para_D + temp_array)

    return sig1, sig2

# Calculates the individual charge carrier concentrations of two bands at one temperature
def dpb_ind_carCon(temp: float, fermi_energy: float, meff: float, gap: float, degen: float) -> tuple[np.ndarray, np.ndarray]:
    
    mu = fsolve(dpb_chemPot, [fermi_energy], (temp,gap,meff,fermi_energy, degen))[0]
    
    carCon1 = spb_carCon(mu, -1, temp) * (-1)
    carCon2 = spb_carCon(mu - gap, meff * degen**(5/3), temp) * np.sign(meff)
    return carCon1, carCon2

# Calculates the individual temperature-dependent charge carrier concentrations of two bands
def dpb_ind_carCon_vs_temp(temp_array: np.ndarray, fermi_energy: float, meff: float, gap: float, degen: float) -> tuple[np.ndarray, np.ndarray]:
    
    chemPot_array = dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)
    charCon1 = np.zeros(shape=len(temp_array))
    charCon2 = np.zeros(shape=len(temp_array))
    
    for i, temp in enumerate(temp_array):   
        charCon1[i] = spb_carCon(chemPot_array[i], -1, temp) * (-1)
        charCon2[i] = spb_carCon(chemPot_array[i] - gap, meff * degen**(5/3), temp) * np.sign(meff)
        print(temp, charCon2[i])
    return charCon1, charCon2


'''
#################################################
Temporary helper functions
#################################################
'''

def parabolic_band(k, mass, energy):
    return((k / 2*np.pi)**2 / (2 * mass) + energy)  

def spart_2PB(T: float, mass: float, gap: float, degen: float, fermi: float) -> list[float]: #args is supposed to look like: mass1,mass2,....,gap1,gap2,.....,fermienergy):
    #degen is Nv2/Nv1
    
    mu = fermi
    
    onsi1 = onsi_general(mu, -1, 1, -1, T)
    onsi2 = onsi_general(mu-gap, mass, 1, -1, T)
    
    mu = fsolve(dpb_chemPot, [fermi], (T,gap,mass,fermi, degen))[0]
    total_sigma = sigma_general(mu, -1, 1, -1, T) + sigma_general(mu-gap, mass, 1, -1, T)

    see1 = onsi1 / total_sigma
    see2 = onsi2 / total_sigma

    return see1, see2
    
def dpb_chemPot_solve(eta_here, T, gap, mass, EF, Nv):
    return fsolve(dpb_chemPot, [eta_here], (T, gap, mass, EF, Nv))[0]

if __name__ == "__main__":
    # alpha = 2*np.sqrt(2)*e**2*kB/(np.pi**2*hbar**3*me)
    # print(f"alpha = {alpha}")
    
    # times = time.time()
    # for i in range(1000000):
    #     result = fdint_robust(0.5, (i/1000.*6.)-3000., 50) 
    # print(time.time()-times)
    
    # print(4/(3*np.sqrt(np.pi)))
    # print(np.pi**2/6.)
    
    # print(fdint_robust(0.5,1.299,1))
    # print(fdint_robust(0.5,1.3,1))
    # print(fdint_robust(0.5,1.301,1))
    # print(fdint_robust(0.5,1.31,1))
    fermi_int_approx_x = []
    fermi_int_approx_y = []
    fermi_int_accur_x = []
    fermi_int_accur_y = []
    for i in range(-10345,20123,1):
    # for i in range(-8250,-7800,1):
    # for i in range(-8250,-7500,1):
        fermi_int_approx_x.append([i/1151])
        fermi_int_approx_y.append([fdint_robust(0.5,i/1151,1)])
        fermi_int_accur_x.append([i/1151])
        fermi_int_accur_y.append([fdint_robust2(0.5,i/1151,1)])
    
    import matplotlib.pyplot as plt
    
    
    # Create a plot

    plt.figure(figsize=(10, 6))
    # plt.plot(fermi_int_approx_x, fermi_int_approx_y, marker='o', linestyle='-', color='b', label='Approximation')
    # plt.plot(fermi_int_accur_x, fermi_int_accur_y, marker='x', linestyle='-', color='g', label='Accurate')
    plt.plot(np.array(fermi_int_accur_x), (np.array(fermi_int_accur_y) - np.array(fermi_int_approx_y))/np.array(fermi_int_accur_y), marker='x', linestyle='-', color='g', label='deviation')
    
    # plt.xlim(708, 710)
    # plt.ylim(fdint_robust(1,708,1), fdint_robust(1,710,1))
    
    # for i in range(1750,2200,1):
    #     # print(i, fermi_int_approx_y[i], fermi_int_accur_y[i])
    #     print(int((i-10000)/1000.*1000+10000), (i-10000)/1000.*1000+10000)
    # Add title and labels
    plt.title('Plot of y vs. x')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    plt.show()
    # f = open("fermi_int_-10_10.txt","w")
    # for i in range(len(fermi_int_accur_x)):
    #     f.write((f"{fermi_int_accur_x[i][0]}\t{fermi_int_accur_y[i][0]:.10e}\n"))
    # for i in range(len(fermi_int_accur_x)):
    #     print(f"{fermi_int_accur_x[i][0]}\t{fermi_int_accur_y[i][0]}")
    # f.close()
    
    # print(fdint_robust(1,710,1))
    # print(fdint_robust2(1,711,1))
    # print(fdint_robust(0.5,0,1))
    # print(fdint_robust(0.5,200,1))
    # print(fdint_robust(0.5,1.31,1))
    
    # print(fdint_robust2(0.5,1,1))
    # print(fdint_robust2(0.5,0.3,1))
    # print(fdint_robust2(0.5,200.301,1))
    # print(fdint_robust2(0.5,1.31,1))
    # print(mpmath.gamma(float(1.5)))
    
    
    # print(fdint_robust2(0.5,1.2,1))
    # print(fdint_robust2(0.5,1.21, 1))
    # print(fdint_robust2(0.5,1.301,1))
    # print(fdint_robust2(0.5,1.31,1))