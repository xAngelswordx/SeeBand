from modules import math_module1 as te
import numpy as np
from scipy import optimize as opt
# from scipy.optimize import fsolve

class PB_fit:
    def __init__(self):
        pass
    
    def is_accuracy_reached(self, threshold, *errors):
        # Check if the error of all fitting parameters is smaller than threshold
        for error in errors:
            if error > threshold:
                return False
        return True



    '''
    #################################################
    Functions to fit the temperature-dependent thermoelectric properties based on the model and mode
    #################################################
    '''

    def spb_see_fit(
        self, 
        temp_list: list[float],
        see_list: list[float],
        band_mass: int,
        initial_fermi_energy: float,
        limits_fermi_energy: tuple[list[float], list[float]],               
        print_info: bool = False
        ):
        
        """Fits the Seebeck coefficient within the single-parabolic-band model and returns the position of the Fermi energy
    
        Parameters
        ----------
        temp_list : numpy.ndarray
            Temperatures of the measured Seebeck coefficient in K
        see_list : numpy.ndarray
            Seebeck coefficient in V/K at temperatures given by temp_list
        band_mass : int
            Mass of the band
        initial_fermi_energy : list[float]
            Initial Fermi energy for the fit
        limits_fermi_energy : tuple(list[float], list[float])
            Lower and upper limits of the Fermi energy for the fit
        print_info : str, optional
            Whether to print the fit parameters and meta information to the terminal
            
        Returns
        -------
        out : list[float]
            List containing the Fermi energy
        """
        
        if print_info:
            print("Running 'spb_see_fit'")

        # Fit the Seebeck coefficient with the parameters 'mass', 'gap' and 'fermi_energy'
        parameter_see, _ = opt.curve_fit(
            lambda T, fermi_energy: te.spb_see_vs_temp(temp_list, fermi_energy, band_mass, scal_fac = 1e6), 
            temp_list, see_list * 1e6, p0 = initial_fermi_energy, bounds = limits_fermi_energy, full_output=False, maxfev=1000)

        # Save the current band parameters
        current_fermi_energy = parameter_see[0]
        
        if print_info:
            print(f"EF: {current_fermi_energy:.5f}\n")
            
        if print_info:
            print("Finished 'spb_see_fit'")

        parameters = [current_fermi_energy]
        return parameters
    
    def dpb_see_fit(
        self, 
        temp_list: list[float],
        see_list: list[float],
        initial_parameter: list[float],
        limits: tuple[list[float], list[float]],               
        degen: float,
        print_info: bool = False
        ):
        
        """Fits the Seebeck coefficient within the double-parabolic-band model and returns band-structure parameters
    
        Parameters
        ----------
        temp_list : numpy.ndarray
            Temperatures of the measured Seebeck coefficient in K
        see_list : numpy.ndarray
            Seebeck coefficient in V/K at temperatures given by temp_list
        initial_parameter : list[float]
            Initial fitting parameters of the Sebeeck coefficient
        limits : tuple(list[float], list[float])
            Lower and upper limits of the fitting parameters of the Seebeck coefficient
        degen : float
            Ratio of the degeneracy of band 2 to band 1       
        print_info : str, optional
            Whether to print the fit parameters and meta information to the terminal
            
        Returns
        -------
        out : list[float]
            List containing the effective mass ratio, band gap and Fermi energy
        """
        
        if print_info:
            print("Running 'dpb_see_fit'")

        # Fit the Seebeck coefficient with the parameters 'mass', 'gap' and 'fermi_energy'
        parameter_see, _ = opt.curve_fit(
            lambda T, meff, gap, fermi_energy: te.dpb_see_vs_temp(temp_list, meff, gap, fermi_energy, degen, scal_fac = 1e6), 
            temp_list, see_list * 1e6, p0 = initial_parameter, bounds = limits, full_output=False, maxfev=1000)

        # Save the current band parameters
        current_meff = parameter_see[0]/degen
        current_gap = parameter_see[1]
        current_fermi_energy = parameter_see[2]
        
        if print_info:
            print(f"Mass: {current_meff:.5f}\n"
                    f"Gap: {current_gap:.5f}\n"
                    f"EF: {current_fermi_energy:.5f}\n")
            
        if print_info:
            print("Finished 'dpb_see_fit'")

        parameters = [current_meff*degen, current_gap, current_fermi_energy]
        return parameters

    def spb_see_res_fit(
        self,
        temp_see_list: np.ndarray,
        see_list: np.ndarray,
        temp_res_list: np.ndarray,
        res_list: np.ndarray,
        band_mass: int,
        initial_fermi_energy: list[float],
        limits_fermi_energy: tuple[list[float], list[float]], 
        initial_parameter_res: list[float],
        limits_res: tuple[list[float], list[float]],
        scatter_type: str,
        print_info: bool = False,
        ) -> list[float]:
        """Fits the Seebeck coefficient and resistivity data within the single-parabolic-band model and returns the Fermi energy and scattering parameters
    
        Parameters
        ----------
        temp_see_list : numpy.ndarray
            Temperatures of the measured Seebeck coefficient in K
        see_list : numpy.ndarray
            Seebeck coefficient in V/K at temperatures given by temp_see_list
        temp_res_list : numpy.ndarray
            Temperatures of the measured resistivity in K
        res_list : float
            Resistivity in Ohm*m at the temperatures given by temp_res_list
        band_mass : int
            Mass of the band
        initial_fermi_energy : list[float]
            Initial Fermi energy for the fit
        limits_fermi_energy : tuple(list[float], list[float])
            Lower and upper limits of the Fermi energy for the fit
        initial_parameter_res : list[float]
            Initial fitting parameters of the resistivity
        limits_res : tuple(list[float], list[float])
            Lower and upper limits of the fitting parameters of the resistivity
        scatter_type : string
            Determines the predominant scattering mechanisms, must be either 'acPh' (acoustic-phonon scattering), 'dis' (alloy-disorder scattering) or 'acPhDis' (both)    
        print_info : str, optional
            Whether to print the fit parameters and meta information to the terminal
            
        Returns
        -------
        out : list[float]
            List containing the effective Fermi energy, scattering parameter A, scattering parameter B, and, in case of acoustic-phonon + alloy-disorder scattering, scattering parameter C and scattering parameter D
        """
        
        if print_info:  
            print("Running 'spb_see_res_fit'")
            
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'spb_see_res_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        parameters = initial_fermi_energy + initial_parameter_res
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del initial_parameter_res[1:3]
            del limits_res[0][1:3]
            del limits_res[1][1:3]
        elif scatter_type == "dis":
            del initial_parameter_res[0:2]
            del limits_res[0][0:2]
            del limits_res[1][0:2]
        else:
            del initial_parameter_res[-1]
            del limits_res[0][-1]
            del limits_res[1][-1]
            initial_parameter_res[1] = initial_parameter_res[1]*300. #adjusted, as the input is given at 300 K
            limits_res[0][1] = limits_res[0][1]*300. #adjusted, as the input is given at 300 K
            limits_res[1][1] = limits_res[1][1]*300. #adjusted, as the input is given at 300 K

            
        
        # Fit the Seebeck coefficient with the Fermi energy
        parameter_see, _ = opt.curve_fit(
            lambda T, fermi_energy: te.spb_see_vs_temp(temp_see_list, fermi_energy, band_mass, scal_fac = 1e6), 
            temp_see_list, see_list * 1e6, p0 = initial_fermi_energy, bounds = limits_fermi_energy, full_output=False, maxfev=1000)

        # Save the current band parameters
        current_fermi_energy = parameter_see[0]
        
        if print_info:
            print(f"EF: {current_fermi_energy:.5f}\n")
            
        if print_info:
                print("Starting Resistivity fit...")
                     
        # Calculate the Integral
        f0 = te.spb_fermiInt_vs_temp(s = 0, T = temp_res_list, mass_sign = np.sign(band_mass), fermi_energy = current_fermi_energy)

        # Fit the resistivity
        if scatter_type == "acPh":
            parameter_res, _ = opt.curve_fit(
                lambda T, para_A: te.spb_rho_vs_temp_acPh(T, f0, para_A, scal_fac = 1e8), 
                temp_res_list, res_list * 1e8, p0 = initial_parameter_res, bounds = limits_res, full_output=False, maxfev=50_000)
        elif scatter_type == "dis":
            parameter_res, _ = opt.curve_fit(
                lambda T, para_A: te.spb_rho_vs_temp_dis(T, f0, para_A, scal_fac = 1e8), 
                temp_res_list, res_list * 1e8, p0 = initial_parameter_res, bounds = limits_res, full_output=False, maxfev=50_000)
        else:
            parameter_res, _ = opt.curve_fit(
                lambda T, para_A, para_B: te.spb_rho_vs_temp_acPh_dis(T, f0, para_A, para_B, scal_fac = 1e8), 
                temp_res_list, res_list * 1e8, p0 = initial_parameter_res, bounds = limits_res, full_output=False, maxfev=50_000)
        
        # Save the current scattering parameters
        current_para_A = parameter_res[0]
        if scatter_type == "acPhDis":
            current_para_B = parameter_res[1]

        if print_info:
            # Print information about the fitting parameters of the resistivity
            if scatter_type == "acPh" or scatter_type == "dis":
                print(f"current_para_A: {current_para_A:.5f}\n")
            else:
                print(f"current_para_A: {current_para_A:.5f}\n"
                    f"current_para_B: {current_para_B:.5f}\n")
        
        
        if print_info:
            print("Finished 'spb_see_res_fit'")

        # Save all fitting parameters in a list and return the list
        parameters[0] = current_fermi_energy
        if scatter_type == "acPh":
            parameters[1] = current_para_A
        elif scatter_type == "dis":
            parameters[3] = current_para_A
        else:
            parameters[1:3] = [current_para_A, current_para_B/300.]
                
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     parameters = [current_fermi_energy, current_para_A]
        # else:
        #     parameters = [current_fermi_energy, current_para_A, current_para_B]
        return parameters

    def dpb_see_res_fit(
        self,
        temp_see_list: np.ndarray,
        see_list: np.ndarray,
        temp_res_list: np.ndarray,
        res_list: np.ndarray,
        initial_parameter_see: list[float],
        limits_see: tuple[list[float], list[float]], 
        initial_parameter_res: list[float],
        limits_res: tuple[list[float], list[float]],
        degen: float,
        scatter_type: str,
        threshold: float,
        max_num_iterations: int,
        print_info: bool = False,
        ) -> tuple[float, float, float, float, float, float, float]:
        """Fits the Seebeck coefficient and resistivity data within the double-parabolic-band model and returns band-structure and scattering parameters
    
        Parameters
        ----------
        temp_see_list : numpy.ndarray
            Temperatures of the measured Seebeck coefficient in K
        see_list : numpy.ndarray
            Seebeck coefficient in V/K at temperatures given by temp_see_list
        temp_res_list : numpy.ndarray
            Temperatures of the measured resistivity in K
        res_list : float
            Resistivity in Ohm*m at the temperatures given by temp_res_list
        initial_parameter_see : list[float]
            Initial fitting parameters of the Sebeeck coefficient
        limits_see : tuple(list[float], list[float])
            Lower and upper limits of the fitting parameters of the Seebeck coefficient
        initial_parameter_res : list[float]
            Initial fitting parameters of the resistivity
        limits_res : tuple(list[float], list[float])
            Lower and upper limits of the fitting parameters of the resistivity
        degen : float
            Ratio of the degeneracy of band 2 to band 1       
        scatter_type : string
            Determines the predominant scattering mechanisms, must be either 'acPh' (acoustic-phonon scattering), 'dis' (alloy-disorder scattering) or 'acPhDis' (both)
        threshold : float
            Upper limit of the relative error of the fitting parameter of the resistivity before the fit stops
        max_num_iterations : int
            Maximum number of iterations before the fit stops
        print_info : str, optional
            Whether to print the fit parameters and meta information to the terminal
            
        Returns
        -------
        out : list[float]
            List containing the effective mass, band gap, Fermi energy, scattering parameter A, scattering parameter B, scattering parameter C and scattering parameter D
        """
        
        if print_info:  
            print("Running 'dpb_see_res_fit'")
            
    
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_see_res_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        parameters = initial_parameter_see + initial_parameter_res
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del initial_parameter_res[2:6]
            del limits_res[0][2:6]
            del limits_res[1][2:6]
        elif scatter_type == "dis":
            del initial_parameter_res[0:4]
            del limits_res[0][0:4]
            del limits_res[1][0:4]
        else:
            del initial_parameter_res[4:6]
            del limits_res[0][4:6]
            del limits_res[1][4:6]
            for i in range(2,4):
                initial_parameter_res[i] = initial_parameter_res[i]*300. #adjusted, as the input is given at 300 K
                limits_res[0][i] = limits_res[0][i]*300. #adjusted, as the input is given at 300 K
                limits_res[1][i] = limits_res[1][i]*300. #adjusted, as the input is given at 300 K
        
        # Set the start paramters of the first setp to the provided initial paramters
        start_parameter_see = initial_parameter_see
        start_paramter_res = initial_parameter_res
        
        # Define some variables...
        current_para_A = start_paramter_res[0]
        current_para_B = start_paramter_res[1]
        prev_para_A = current_para_A
        prev_para_B = current_para_B
        if scatter_type == "acPhDis":
            current_para_C = start_paramter_res[2]
            current_para_D = start_paramter_res[3]
            prev_para_C = current_para_C
            prev_para_D = current_para_D
        current_meff = start_parameter_see[0]/degen
        current_gap = start_parameter_see[1]
        current_fermi_energy = start_parameter_see[2]
        prev_meff = current_meff
        prev_gap = current_gap
        prev_fermi_energy = current_fermi_energy
        
        parameter_res = start_paramter_res
        parameter_see = start_parameter_see
        
        # Save the number of iterations to track the progress
        number_iterations = 1
        
        # Infinite loop that ends if accuracy is reached or number of iterations is too large
        while True:
            if print_info:
                # print the current number of iterations
                print(f"Number of iteration: {number_iterations}")
                print("Starting Seebeck fit...")
                
            # Fit the Seebeck coefficient with the parameters 'mass', 'gap' and 'fermi_energy'
            if scatter_type == "acPh" or scatter_type == "dis":
                parameter_see, _ = opt.curve_fit(
                    lambda T, mass, gap, fermi_energy: te.dpb_see_vs_temp_single_scatter(temp_see_list, mass, gap, fermi_energy, degen, current_para_B, scal_fac = 1e6), 
                    temp_see_list, see_list * 1e6, p0 = start_parameter_see, bounds = limits_see, full_output=False, maxfev=1000)
            else:
                parameter_see, _ = opt.curve_fit(
                    lambda T, mass, gap, fermi_energy: te.dpb_see_vs_temp_double_scatter(temp_see_list, mass, gap, fermi_energy, degen, current_para_B, current_para_C, current_para_D, scal_fac = 1e6), 
                    temp_see_list, see_list * 1e6, p0 = start_parameter_see, bounds = limits_see, full_output=False, maxfev=1000)
            
            # Save the current band parameters
            current_meff = parameter_see[0]
            current_gap = parameter_see[1]
            current_fermi_energy = parameter_see[2]
            
            # Set the start parameters of the Sebeeck coefficient fit to the currently best values
            start_parameter_see = parameter_see
            
            if print_info:
                print(f"Mass: {current_meff:.5f}\n"
                      f"Gap: {current_gap:.5f}\n"
                      f"EF: {current_fermi_energy:.5f}\n")
                
            if print_info:
                print("Starting Resistivity fit...")
                     
            # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
            f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s = 0, T = temp_res_list, mass = current_meff, gap = current_gap, fermi_energy = current_fermi_energy, degen = degen)

            # Check if number of iterations is <= max_num_iterations to ensure that the loop is not infinite
            # If the fit is not converged after max_num_iterations steps, the resistivity is fitted one more time while keeping the factor for the Seebeck coefficient fixed
            if number_iterations > max_num_iterations:
                if print_info:
                    print(f"Error! Fitting not possible. Number of iterations in 'dpb_see_res_fit' exceeds {max_num_iterations}")                
                break

            # Fit the resistivity with the parameters 'para_A', para_B', 'para_C' and 'para_D'
            if scatter_type == "acPh":
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B: te.dpb_rho_vs_temp_acPh(T, f0_1, f0_2, para_A, para_B, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            elif scatter_type == "dis":
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B: te.dpb_rho_vs_temp_dis(T, f0_1, f0_2, para_A, para_B, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            else:
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B, para_C, para_D: te.dpb_rho_vs_temp_acPh_dis(T, f0_1, f0_2, para_A, para_B, para_C, para_D, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            
            # Save the current scattering parameters
            current_para_A = parameter_res[0]
            current_para_B = parameter_res[1]
            if scatter_type == "acPhDis":
                current_para_C = parameter_res[2]
                current_para_D = parameter_res[3]
            
            # Set the start parameters of the resistivity fit to the currently best values
            start_paramter_res = parameter_res

            if print_info:
                # Print information about the fitting parameters of the resistivity
                if scatter_type == "acPh" or scatter_type == "dis":
                    print(f"current_para_A: {current_para_A:.5f}\n"
                        f"current_para_B: {current_para_B:.5f}\n")
                else:
                    print(f"current_para_A: {current_para_A:.5f}\n"
                        f"current_para_B: {current_para_B:.5f}\n"
                        f"current_para_C: {current_para_C:.5f}\n"
                        f"current_para_D: {current_para_D:.5f}")

            # Calculate the error of the fitting parameters of the resistivtiy
            error_meff         = np.abs((current_meff - prev_meff)/prev_meff)
            error_gap          = np.abs((current_gap - prev_gap)/prev_gap)
            error_fermi_energy = np.abs((current_fermi_energy - prev_fermi_energy)/prev_fermi_energy)
            error_para_A       = np.abs((current_para_A - prev_para_A)/prev_para_A)
            error_para_B       = np.abs((current_para_B - prev_para_B)/prev_para_B)
            if scatter_type == "acPhDis":
                error_para_C   = np.abs((current_para_C - prev_para_C)/prev_para_C)
                error_para_D   = np.abs((current_para_D - prev_para_D)/prev_para_D)
            
            # Check if the error of all fitting parameters is smaller than threshold
            if self.is_accuracy_reached(threshold, error_meff, error_gap, error_fermi_energy, error_para_A, error_para_B):
                if print_info:
                    print("Accuracy reached.")
                break
            if scatter_type == "acPhDis" and self.is_accuracy_reached(threshold, error_para_C, error_para_D):
                if print_info:
                    print("Accuracy reached.")
                break
            
            # Save the current fitting parameters to determine the error in the next step
            prev_meff = current_meff
            prev_gap = current_gap
            prev_fermi_energy = current_fermi_energy
            prev_para_A = current_para_A
            prev_para_B = current_para_B
            if scatter_type == "acPhDis":
                prev_para_C = current_para_C
                prev_para_D = current_para_D
            
            # Increase the number of iterations by 1
            number_iterations += 1
            
        # Print the number of steps needed to reach the desired accuracy
        if print_info:
            print(f"\nIt took {number_iterations} iterations to evaluate the best fit\n")
        
        if print_info:           
            print("###########################################")
            print("Band structure parameters:")
            print(
                f"Mass: {current_meff:.5f}\n"
                f"Gap: {current_gap:.5f}\n"
                f"EF: {current_fermi_energy:.5f}\n"
                )
            print("Scattering parameters:")
            if scatter_type == "acPh" or scatter_type == "dis":
                print(
                    f"parameter A: {current_para_A:.3f}\n"
                    f"parameter B: {current_para_B:.3f}\n"
                    )
            else:
                print(
                    f"parameter A: {current_para_A:.3f}\n"
                    f"parameter B: {current_para_B:.3f}\n"
                    f"parameter C: {current_para_C:.3f}\n"
                    f"parameter D: {current_para_D:.3f}\n"
                    )
            print("###########################################")
        
        print("Finished 'dpb_see_res_fit'")
        
        # Save all fitting parameters in a list and return the list
        parameters[0:3] = [current_meff*degen, current_gap, current_fermi_energy]
        if scatter_type == "acPh":
            parameters[3:5] = [current_para_A, current_para_B]
        elif scatter_type == "dis":
            parameters[7:9] = [current_para_A, current_para_B]
        else:
            parameters[3:7] = [current_para_A, current_para_B, current_para_C/300., current_para_D/300.] #adjusted, as the input is given at 300 K
        
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     parameters = [current_meff, current_gap, current_fermi_energy, current_para_A, current_para_B]
        # else:
        #     parameters = [current_meff, current_gap, current_fermi_energy, current_para_A, current_para_B, current_para_C, current_para_D]
        return parameters
    
    def spb_see_res_hall_fit(
        self,
        temp_see_list: np.ndarray,
        see_list: np.ndarray,
        temp_res_list: np.ndarray,
        res_list: np.ndarray,
        temp_hall_list: np.ndarray,
        hall_list: np.ndarray,
        initial_fermi_energy: list[float],
        limits_fermi_energy: tuple[list[float], list[float]], 
        initial_parameter_res: list[float],
        limits_res: tuple[list[float], list[float]],
        initial_mass1: list[float],
        limits_mass1: tuple[list[float], list[float]],
        scatter_type: str,
        print_info: bool = False,
        ) -> list[float]:
        """Fits the Seebeck coefficient, resistivity and Hall coefficient data within the single-parabolic-band model and returns the Fermi energy, scattering parameters and the band mass
    
        Parameters
        ----------
        temp_see_list : numpy.ndarray
            Temperatures of the measured Seebeck coefficient in K
        see_list : numpy.ndarray
            Seebeck coefficient in V/K at temperatures given by temp_see_list
        temp_res_list : numpy.ndarray
            Temperatures of the measured resistivity in K
        res_list : numpy.ndarray
            Resistivity in Ohm*m at the temperatures given by temp_res_list
        temp_hall_list : numpy.ndarray
            Temperatures of the measured Hall coefficient in K
        hall_list : numpy.ndarray
            Hall coefficient in m^3/(As) at temperatures given by temp_hall_list
        initial_fermi_energy : list[float]
            Initial Fermi energy for the fit
        limits_fermi_energy : tuple(list[float], list[float])
            Lower and upper limits of the Fermi energy for the fit
        initial_parameter_res : list[float]
            Initial fitting parameters of the resistivity
        limits_res : tuple(list[float], list[float])
            Lower and upper limits of the fitting parameters of the resistivity
        initial_mass1 : list[float]
            Initial mass of band 1 for the fit
        limits_mass1 : tuple[list[float], list[float]]
            Lower and upper limits of the mass of band 1 for the fit
        scatter_type : string
            Determines the predominant scattering mechanisms, must be either 'acPh' (acoustic-phonon scattering), 'dis' (alloy-disorder scattering) or 'acPhDis' (both)    
        print_info : str, optional
            Whether to print the fit parameters and meta information to the terminal
            
        Returns
        -------
        out : list[float]
            List containing the effective Fermi energy, scattering parameter A, scattering parameter B, and, in case of acoustic-phonon + alloy-disorder scattering, scattering parameter C and scattering parameter D, and parameter E
        """

        initial_para_E = [np.abs(initial_mass1[0])**(3/2)]*-np.sign(initial_mass1[0]) #!!! Hotfix for the sign, this should be clarified for the final version!
        limits_para_E = [np.abs(limits_mass1[0][0])**(3/2)*np.sign(limits_mass1[0][0])],[np.abs(limits_mass1[1][0])**(3/2)*np.sign(limits_mass1[1][0])] #!!! Hotfix for the sign, this should be clarified for the final version!
        
        band_mass = initial_mass1[0]
        
        if print_info:  
            print("Running 'spb_see_res_hall_fit'")
            
        # Check if scatter_type is properly set to acPh, dis or acPhAlDi
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'spb_see_res_hall_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # # Check if length of initial_parameter_res and limits_res is according to scatter_type
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     if len(initial_parameter_res) != 1:
        #         raise Exception(f"Error! Parameter 'initial_parameter_res' in function 'spb_see_res_hall_fit' must have length 1 for scatter_type: {scatter_type}")
        #     elif len(limits_res[0]) != 1 or len(limits_res[1]) != 1:
        #         raise Exception(f"Error! Parameter 'limits_res' in function 'spb_see_res_hall_fit' must be tuple with arrays of length 1 for scatter_type: {scatter_type}")
        # else:
        #     if len(initial_parameter_res) != 2:
        #         raise Exception(f"Error! Parameter 'initial_parameter_res' in function 'spb_see_res_hall_fit' must have length 2 for scatter_type: {scatter_type}")
        #     elif len(limits_res[0]) != 2 or len(limits_res[1]) != 2:
        #         raise Exception(f"Error! Parameter 'limits_res' in function 'spb_see_res_hall_fit' must be tuple with arrays of length 2 for scatter_type: {scatter_type}")
        
        parameters = initial_fermi_energy + initial_parameter_res + initial_para_E
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del initial_parameter_res[1:3]
            del limits_res[0][1:3]
            del limits_res[1][1:3]
        elif scatter_type == "dis":
            del initial_parameter_res[0:2]
            del limits_res[0][0:2]
            del limits_res[1][0:2]
        else:
            del initial_parameter_res[-1]
            del limits_res[0][-1]
            del limits_res[1][-1]
            initial_parameter_res[1] = initial_parameter_res[1]*300. #adjusted, as the input is given at 300 K
            limits_res[0][1] = limits_res[0][1]*300. #adjusted, as the input is given at 300 K
            limits_res[1][1] = limits_res[1][1]*300. #adjusted, as the input is given at 300 K
            
        # Fit the Seebeck coefficient with the Fermi energy
        parameter_see, _ = opt.curve_fit(
            lambda T, fermi_energy: te.spb_see_vs_temp(temp_see_list, fermi_energy, band_mass, scal_fac = 1e6), 
            temp_see_list, see_list * 1e6, p0 = initial_fermi_energy, bounds = limits_fermi_energy, full_output=False, maxfev=1000)

        # Save the current band parameters
        current_fermi_energy = parameter_see[0]
        
        if print_info:
            print(f"EF: {current_fermi_energy:.5f}\n")
            
        if print_info:
                print("Starting Resistivity fit...")
                     
        # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0 = te.spb_fermiInt_vs_temp(s = 0, T = temp_res_list, mass_sign = np.sign(band_mass), fermi_energy = current_fermi_energy)

        # Fit the resistivity
        if scatter_type == "acPh":
            parameter_res, _ = opt.curve_fit(
                lambda T, para_A: te.spb_rho_vs_temp_acPh(T, f0, para_A, scal_fac = 1e8), 
                temp_res_list, res_list * 1e8, p0 = initial_parameter_res, bounds = limits_res, full_output=False, maxfev=50_000)
        elif scatter_type == "dis":
            parameter_res, _ = opt.curve_fit(
                lambda T, para_A: te.spb_rho_vs_temp_dis(T, f0, para_A, scal_fac = 1e8), 
                temp_res_list, res_list * 1e8, p0 = initial_parameter_res, bounds = limits_res, full_output=False, maxfev=50_000)
        else:
            parameter_res, _ = opt.curve_fit(
                lambda T, para_A, para_B: te.spb_rho_vs_temp_acPh_dis(T, f0, para_A, para_B, scal_fac = 1e8), 
                temp_res_list, res_list * 1e8, p0 = initial_parameter_res, bounds = limits_res, full_output=False, maxfev=50_000)
        
        # Save the current scattering parameters
        current_para_A = parameter_res[0]
        if scatter_type == "acPhDis":
            current_para_B = parameter_res[1]

        if print_info:
            # Print information about the fitting parameters of the resistivity
            if scatter_type == "acPh" or scatter_type == "dis":
                print(f"current_para_A: {current_para_A:.5f}\n")
            else:
                print(f"current_para_A: {current_para_A:.5f}\n"
                    f"current_para_B: {current_para_B:.5f}\n")
        
        if print_info:
            print("Starting Hall coefficient fit...")
            
        # Fit the hall with the parameters 'para_E'
        parameter_hall, _ = opt.curve_fit(
            lambda T, para_E: te.spb_hall_vs_temp(T, current_fermi_energy, np.sign(band_mass), para_E, scal_fac = 1e10), 
            temp_hall_list, hall_list * 1e10, p0 = initial_para_E, bounds = limits_para_E, full_output=False, maxfev=5000)
        
        # Save the current scattering parameters
        current_para_E = parameter_hall[0]
        current_mass1 = current_para_E**(2/3)
        
        if print_info:
            print("Finished 'spb_see_res_hall_fit'")

        # Save all fitting parameters in a list and return the list
        parameters[0] = current_fermi_energy
        parameters[4] = current_mass1
        if scatter_type == "acPh":
            parameters[1] = current_para_A
        elif scatter_type == "dis":
            parameters[3] = current_para_A
        else:
            parameters[1:3] = [current_para_A, current_para_B]
        
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     parameters = [current_fermi_energy, current_para_A, current_mass1]
        # else:
        #     parameters = [current_fermi_energy, current_para_A, current_para_B, current_mass1]
        return parameters
    
    def dpb_see_res_hall_fit(
        self,
        temp_see_list: np.ndarray,
        see_list: np.ndarray,
        temp_res_list: np.ndarray,
        res_list: np.ndarray,
        temp_hall_list: np.ndarray,
        hall_list: np.ndarray,
        initial_parameter_see: list[float],
        limits_see: tuple[list[float], list[float]], 
        initial_parameter_res: list[float],
        limits_res: tuple[list[float], list[float]],
        initial_mass1: float,
        limits_mass1: tuple[list[float], list[float]],
        degen: float,
        scatter_type: str,
        threshold: float,
        max_num_iterations: int,
        print_info: bool = False,
        ) -> tuple[float, float, float, float, float, float, float]:
        """Fits the Seebeck coefficient, resistivity and Hall coefficient data within the double-parabolic-band model and returns band-structure and scattering parameters
    
        Parameters
        ----------
        temp_see_list : numpy.ndarray
            Temperatures of the measured Seebeck coefficient in K
        see_list : numpy.ndarray
            Seebeck coefficient in V/K at temperatures given by temp_see_list
        temp_res_list : numpy.ndarray
            Temperatures of the measured resistivity in K
        res_list : float
            Resistivity in Ohm*m at the temperatures given by temp_res_list
        temp_hall_list : numpy.ndarray
            Temperatures of the measured Hall coefficient in K
        hall_list : numpy.ndarray
            Hall coefficient in m^3/(As) at temperatures given by temp_hall_list
        initial_parameter_see : list[float]
            Initial fitting parameters of the Sebeeck coefficient
        limits_see : tuple[list[float], list[float]]
            Lower and upper limits of the fitting parameters of the Seebeck coefficient
        initial_parameter_res : list[float]
            Initial fitting parameters of the resistivity
        limits_res : tuple[list[float], list[float]]
            Lower and upper limits of the fitting parameters of the resistivity
        initial_mass1 : float
            Initial mass of band 1 for the fit
        limits_mass1 : tuple[list[float], list[float]]
            Lower and upper limits of the mass of band 1 for the fit
        degen : float
            Ratio of the degeneracy of band 2 to band 1       
        scatter_type : string
            Determines the predominant scattering mechanisms, must be either 'acPh' (acoustic-phonon scattering), 'dis' (alloy-disorder scattering) or 'acPhDis' (both)
        threshold : float
            Upper limit of the relative error of the fitting parameter of the resistivity before the fit stops
        max_num_iterations : int
            Maximum number of iterations before the fit stops
        print_info : str, optional
            Whether to print the fit parameters and meta information to the terminal
            
        Returns
        -------
        out : list[float]
            List containing the effective mass, band gap, Fermi energy, scattering parameter A, scattering parameter B, scattering parameter C and scattering parameter D, and parameter E
        """
        
        initial_para_E = [initial_mass1**(3/2)]
        limits_para_E = [[limits_mass1[0][0]**(3/2)],[limits_mass1[1][0]**(3/2)]]
        
        if print_info:  
            print("Running 'dpb_see_res_hall_fit'")
            
        # Check if scatter_type is properly set to acPh, dis or acPhAlDi
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_see_res_hall_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # # Check if length of initial_parameter_res and limits_res is according to scatter_type
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     if len(initial_parameter_res) != 2:
        #         raise Exception(f"Error! Parameter 'initial_parameter_res' in function 'dpb_see_res_hall_fit' must have length 2 for scatter_type: {scatter_type}")
        #     elif len(limits_res[0]) != 2 or len(limits_res[1]) != 2:
        #         raise Exception(f"Error! Parameter 'limits_res' in function 'dpb_see_res_hall_fit' must be tuple with arrays of length 2 for scatter_type: {scatter_type}")
        # else:
        #     if len(initial_parameter_res) != 4:
        #         raise Exception(f"Error! Parameter 'initial_parameter_res' in function 'dpb_see_res_hall_fit' must have length 4 for scatter_type: {scatter_type}")
        #     elif len(limits_res[0]) != 4 or len(limits_res[1]) != 4:
        #         raise Exception(f"Error! Parameter 'limits_res' in function 'dpb_see_res_hall_fit' must be tuple with arrays of length 2 for scatter_type: {scatter_type}")
        
        parameters = initial_parameter_see + initial_parameter_res + initial_para_E
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del initial_parameter_res[2:6]
            del limits_res[0][2:6]
            del limits_res[1][2:6]
        elif scatter_type == "dis":
            del initial_parameter_res[0:4]
            del limits_res[0][0:4]
            del limits_res[1][0:4]
        else:
            del initial_parameter_res[4:6]
            del limits_res[0][4:6]
            del limits_res[1][4:6]
            for i in range(2,4):
                initial_parameter_res[i] = initial_parameter_res[i]*300. #adjusted, as the input is given at 300 K
                limits_res[0][i] = limits_res[0][i]*300. #adjusted, as the input is given at 300 K
                limits_res[1][i] = limits_res[1][i]*300. #adjusted, as the input is given at 300 K
        
        # Set the start paramters of the first setp to the provided initial paramters
        start_parameter_see = initial_parameter_see
        start_paramter_res = initial_parameter_res
        
        # Define some variables...
        current_para_A = start_paramter_res[0]
        current_para_B = start_paramter_res[1]
        prev_para_A = current_para_A
        prev_para_B = current_para_B
        if scatter_type == "acPhDis":
            current_para_C = start_paramter_res[2]
            current_para_D = start_paramter_res[3]
            prev_para_C = current_para_C
            prev_para_D = current_para_D
        current_meff = start_parameter_see[0]/degen
        current_gap = start_parameter_see[1]
        current_fermi_energy = start_parameter_see[2]
        prev_meff = current_meff
        prev_gap = current_gap
        prev_fermi_energy = current_fermi_energy
        
        parameter_res = start_paramter_res
        parameter_see = start_parameter_see
        
        # Save the number of iterations to track the progress
        number_iterations = 1
        
        # Infinite loop that ends if accuracy is reached or number of iterations is too large
        while True:
            if print_info:
                # print the current number of iterations
                print(f"Number of iteration: {number_iterations}")
                print("Starting Seebeck fit...")
                
            # Fit the Seebeck coefficient with the parameters 'mass', 'gap' and 'fermi_energy'
            if scatter_type == "acPh" or scatter_type == "dis":
                parameter_see, _ = opt.curve_fit(
                    lambda T, mass, gap, fermi_energy: te.dpb_see_vs_temp_single_scatter(temp_see_list, mass, gap, fermi_energy, degen, current_para_B, scal_fac = 1e6), 
                    temp_see_list, see_list * 1e6, p0 = start_parameter_see, bounds = limits_see, full_output=False, maxfev=1000)
            else:
                parameter_see, _ = opt.curve_fit(
                    lambda T, mass, gap, fermi_energy: te.dpb_see_vs_temp_double_scatter(temp_see_list, mass, gap, fermi_energy, degen, current_para_B, current_para_C, current_para_D, scal_fac = 1e6), 
                    temp_see_list, see_list * 1e6, p0 = start_parameter_see, bounds = limits_see, full_output=False, maxfev=1000)
            
            # Save the current band parameters
            current_meff = parameter_see[0]
            current_gap = parameter_see[1]
            current_fermi_energy = parameter_see[2]
            
            # Set the start parameters of the Sebeeck coefficient fit to the currently best values
            start_parameter_see = parameter_see
            
            if print_info:
                print(f"Mass: {current_meff:.5f}\n"
                      f"Gap: {current_gap:.5f}\n"
                      f"EF: {current_fermi_energy:.5f}\n")
                
            if print_info:
                print("Starting Resistivity fit...")
                     
            # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
            f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s = 0, T = temp_res_list, mass = current_meff, gap = current_gap, fermi_energy = current_fermi_energy, degen = degen)

            # Check if number of iterations is <= max_num_iterations to ensure that the loop is not infinite
            # If the fit is not converged after max_num_iterations steps, the resistivity is fitted one more time while keeping the factor for the Seebeck coefficient fixed
            if number_iterations > max_num_iterations:
                if print_info:
                    print(f"Error! Fitting not possible. Number of iterations in 'dpb_see_res_fit' exceeds {max_num_iterations}")                
                break

            # Fit the resistivity with the parameters 'para_A', para_B', 'para_C' and 'para_D'
            if scatter_type == "acPh":
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B: te.dpb_rho_vs_temp_acPh(T, f0_1, f0_2, para_A, para_B, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            elif scatter_type == "dis":
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B: te.dpb_rho_vs_temp_dis(T, f0_1, f0_2, para_A, para_B, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            else:
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B, para_C, para_D: te.dpb_rho_vs_temp_acPh_dis(T, f0_1, f0_2, para_A, para_B, para_C, para_D, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            
            # Save the current scattering parameters
            current_para_A = parameter_res[0]
            current_para_B = parameter_res[1]
            if scatter_type == "acPhDis":
                current_para_C = parameter_res[2]
                current_para_D = parameter_res[3]
            
            # Set the start parameters of the resistivity fit to the currently best values
            start_paramter_res = parameter_res

            if print_info:
                # Print information about the fitting parameters of the resistivity
                if scatter_type == "acPh" or scatter_type == "dis":
                    print(f"current_para_A: {current_para_A:.5f}\n"
                        f"current_para_B: {current_para_B:.5f}\n")
                else:
                    print(f"current_para_A: {current_para_A:.5f}\n"
                        f"current_para_B: {current_para_B:.5f}\n"
                        f"current_para_C: {current_para_C:.5f}\n"
                        f"current_para_D: {current_para_D:.5f}")

            # Calculate the error of the fitting parameters of the resistivtiy
            error_meff         = np.abs((current_meff - prev_meff)/prev_meff)
            error_gap          = np.abs((current_gap - prev_gap)/prev_gap)
            error_fermi_energy = np.abs((current_fermi_energy - prev_fermi_energy)/prev_fermi_energy)
            error_para_A       = np.abs((current_para_A - prev_para_A)/prev_para_A)
            error_para_B       = np.abs((current_para_B - prev_para_B)/prev_para_B)
            if scatter_type == "acPhDis":
                error_para_C   = np.abs((current_para_C - prev_para_C)/prev_para_C)
                error_para_D   = np.abs((current_para_D - prev_para_D)/prev_para_D)
            
            # Check if the error of all fitting parameters is smaller than threshold
            if self.is_accuracy_reached(threshold, error_meff, error_gap, error_fermi_energy, error_para_A, error_para_B):
                if print_info:
                    print("Accuracy reached.")
                break
            if scatter_type == "acPhDis" and self.is_accuracy_reached(threshold, error_para_C, error_para_D):
                if print_info:
                    print("Accuracy reached.")
                break

            # Save the current fitting parameters to determine the error in the next step
            prev_meff = current_meff
            prev_gap = current_gap
            prev_fermi_energy = current_fermi_energy
            prev_para_A = current_para_A
            prev_para_B = current_para_B
            if scatter_type == "acPhDis":
                prev_para_C = current_para_C
                prev_para_D = current_para_D
            
            # Increase the number of iterations by 1
            number_iterations += 1
            
        if print_info:
            print("Starting Hall fit...")
            
        #Calculate Fermi integrals and rho at temperatures of hall
        f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s = 0, T = temp_hall_list, mass=current_meff, gap=current_gap, fermi_energy=current_fermi_energy, degen=degen)
        fm05_1, fm05_2 = te.dpb_fermiInt_vs_temp(s = -0.5, T = temp_hall_list, mass = current_meff, gap = current_gap, fermi_energy=current_fermi_energy, degen=degen)
        
        if scatter_type == "acPh":
            res_array_hall_temp = te.dpb_rho_vs_temp_acPh(temp_hall_list, f0_1, f0_2, current_para_A, current_para_B, current_meff)
        if scatter_type == "dis":
            res_array_hall_temp = te.dpb_rho_vs_temp_dis(temp_hall_list, f0_1, f0_2, current_para_A, current_para_B, current_meff)
        if scatter_type == "acPhDis":
            res_array_hall_temp = te.dpb_rho_vs_temp_acPh_dis(temp_hall_list, f0_1, f0_2, current_para_A, current_para_B, current_para_C, current_para_D, current_meff)
        
        # Fit the hall with the parameter 'para_E'
        if scatter_type == "acPh":
            parameter_hall, _ = opt.curve_fit(
            lambda T, para_E: te.dpb_hall_vs_temp_acPh(T, res_array_hall_temp, fm05_1, fm05_2, current_meff, degen, current_para_A, current_para_B, para_E, scal_fac = 1e10), 
            temp_hall_list, hall_list * 1e10, p0 = initial_para_E, bounds = limits_para_E, full_output=False, maxfev=5000)
        elif scatter_type == "dis":
            parameter_hall, _ = opt.curve_fit(
            lambda T, para_E: te.dpb_hall_vs_temp_dis(T, res_array_hall_temp, fm05_1, fm05_2, current_meff, degen, current_para_A, current_para_B, para_E, scal_fac = 1e10), 
            temp_hall_list, hall_list * 1e10, p0 = initial_para_E, bounds = limits_para_E, full_output=False, maxfev=5000)
        else:
            parameter_hall, _ = opt.curve_fit(
                lambda T, para_E: te.dpb_hall_vs_temp_acPh_dis(T, res_array_hall_temp, fm05_1, fm05_2, current_meff, degen, current_para_A, current_para_B, current_para_C, current_para_D, para_E, scal_fac = 1e10), 
                temp_hall_list, hall_list * 1e10, p0 = initial_para_E, bounds = limits_para_E, full_output=False, maxfev=5000)

        # Save the current scattering parameters
        current_para_E = parameter_hall[0]
        current_mass1 = current_para_E**(2/3)
        
        # Print the number of steps needed to reach the desired accuracy
        if print_info:
            print(f"\nIt took {number_iterations} iterations to evaluate the best fit\n")
        
        if print_info:           
            print("###########################################")
            print("Band structure parameters:")
            print(
                f"Mass: {current_meff:.5f}\n"
                f"Gap: {current_gap:.5f}\n"
                f"EF: {current_fermi_energy:.5f}\n"
                )
            print("Scattering parameters:")
            if scatter_type == "acPh" or scatter_type == "dis":
                print(
                    f"parameter A: {current_para_A:.3f}\n"
                    f"parameter B: {current_para_B:.3f}\n"
                    )
            else:
                print(
                    f"parameter A: {current_para_A:.3f}\n"
                    f"parameter B: {current_para_B:.3f}\n"
                    f"parameter C: {current_para_C:.3f}\n"
                    f"parameter D: {current_para_D:.3f}\n"
                    )
            print(
                f"parameter E: {current_para_E}\n"
                f"band mass 1: {current_mass1}\n"
            )
            print("###########################################")
        
        print("Finished 'dpb_see_res_hall_fit'")
        
        # Save all fitting parameters in a list and return the list
        parameters[0:3] = [current_meff*degen, current_gap, current_fermi_energy]
        parameters[9] = current_mass1
        if scatter_type == "acPh":
            parameters[3:5] = [current_para_A, current_para_B]
        elif scatter_type == "dis":
            parameters[7:9] = [current_para_A, current_para_B]
        else:
            parameters[3:7] = [current_para_A, current_para_B, current_para_C/300., current_para_D/300.] #adjusted, as the input is given at 300 K

        
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     parameters = [current_meff, current_gap, current_fermi_energy, current_para_A, current_para_B, current_mass1]
        # else:
        #     parameters = [current_meff, current_gap, current_fermi_energy, current_para_A, current_para_B, current_para_C, current_para_D, current_mass1]
        return parameters
    
    # New fit with meff consideration in resistivity fit
    def dpb_see_res_hall_fit_new(
        self,
        temp_see_list: np.ndarray,
        see_list: np.ndarray,
        temp_res_list: np.ndarray,
        res_list: np.ndarray,
        temp_hall_list: np.ndarray,
        hall_list: np.ndarray,
        initial_parameter_see: list[float],
        limits_see: tuple[list[float], list[float]], 
        initial_parameter_res: list[float],
        limits_res: tuple[list[float], list[float]],
        initial_mass1: float,
        limits_mass1: tuple[list[float], list[float]],
        degen: float,
        scatter_type: str,
        threshold: float,
        max_num_iterations: int,
        print_info: bool = False,
        ) -> tuple[float, float, float, float, float, float, float]:
        """Fits the Seebeck coefficient, resistivity and Hall coefficient data within the double-parabolic-band model and returns band-structure and scattering parameters
    
        Parameters
        ----------
        temp_see_list : numpy.ndarray
            Temperatures of the measured Seebeck coefficient in K
        see_list : numpy.ndarray
            Seebeck coefficient in V/K at temperatures given by temp_see_list
        temp_res_list : numpy.ndarray
            Temperatures of the measured resistivity in K
        res_list : float
            Resistivity in Ohm*m at the temperatures given by temp_res_list
        temp_hall_list : numpy.ndarray
            Temperatures of the measured Hall coefficient in K
        hall_list : numpy.ndarray
            Hall coefficient in m^3/(As) at temperatures given by temp_hall_list
        initial_parameter_see : list[float]
            Initial fitting parameters of the Sebeeck coefficient
        limits_see : tuple[list[float], list[float]]
            Lower and upper limits of the fitting parameters of the Seebeck coefficient
        initial_parameter_res : list[float]
            Initial fitting parameters of the resistivity
        limits_res : tuple[list[float], list[float]]
            Lower and upper limits of the fitting parameters of the resistivity
        initial_mass1 : float
            Initial mass of band 1 for the fit
        limits_mass1 : tuple[list[float], list[float]]
            Lower and upper limits of the mass of band 1 for the fit
        degen : float
            Ratio of the degeneracy of band 2 to band 1       
        scatter_type : string
            Determines the predominant scattering mechanisms, must be either 'acPh' (acoustic-phonon scattering), 'dis' (alloy-disorder scattering) or 'acPhDis' (both)
        threshold : float
            Upper limit of the relative error of the fitting parameter of the resistivity before the fit stops
        max_num_iterations : int
            Maximum number of iterations before the fit stops
        print_info : str, optional
            Whether to print the fit parameters and meta information to the terminal
            
        Returns
        -------
        out : list[float]
            List containing the effective mass, band gap, Fermi energy, scattering parameter A, scattering parameter B, scattering parameter C and scattering parameter D, and parameter E
        """
        
        initial_para_E = [initial_mass1**(3/2)]
        limits_para_E = [[limits_mass1[0][0]**(3/2)],[limits_mass1[1][0]**(3/2)]]
        
        if print_info:  
            print("Running 'dpb_see_res_hall_fit'")
            
        # Check if scatter_type is properly set to acPh, dis or acPhAlDi
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_see_res_hall_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # # Check if length of initial_parameter_res and limits_res is according to scatter_type
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     if len(initial_parameter_res) != 2:
        #         raise Exception(f"Error! Parameter 'initial_parameter_res' in function 'dpb_see_res_hall_fit' must have length 2 for scatter_type: {scatter_type}")
        #     elif len(limits_res[0]) != 2 or len(limits_res[1]) != 2:
        #         raise Exception(f"Error! Parameter 'limits_res' in function 'dpb_see_res_hall_fit' must be tuple with arrays of length 2 for scatter_type: {scatter_type}")
        # else:
        #     if len(initial_parameter_res) != 4:
        #         raise Exception(f"Error! Parameter 'initial_parameter_res' in function 'dpb_see_res_hall_fit' must have length 4 for scatter_type: {scatter_type}")
        #     elif len(limits_res[0]) != 4 or len(limits_res[1]) != 4:
        #         raise Exception(f"Error! Parameter 'limits_res' in function 'dpb_see_res_hall_fit' must be tuple with arrays of length 2 for scatter_type: {scatter_type}")
        
        parameters = initial_parameter_see + initial_parameter_res + initial_para_E
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del initial_parameter_res[2:6]
            del limits_res[0][2:6]
            del limits_res[1][2:6]
        elif scatter_type == "dis":
            del initial_parameter_res[0:4]
            del limits_res[0][0:4]
            del limits_res[1][0:4]
        else:
            del initial_parameter_res[4:6]
            del limits_res[0][4:6]
            del limits_res[1][4:6]
            for i in range(2,4):
                initial_parameter_res[i] = initial_parameter_res[i]*300. #adjusted, as the input is given at 300 K
                limits_res[0][i] = limits_res[0][i]*300. #adjusted, as the input is given at 300 K
                limits_res[1][i] = limits_res[1][i]*300. #adjusted, as the input is given at 300 K
        
        # Set the start paramters of the first setp to the provided initial paramters
        start_parameter_see = initial_parameter_see
        start_paramter_res = initial_parameter_res
        
        # Define some variables...
        current_para_A = start_paramter_res[0]
        current_para_B = start_paramter_res[1]
        prev_para_A = current_para_A
        prev_para_B = current_para_B
        if scatter_type == "acPhDis":
            current_para_C = start_paramter_res[2]
            current_para_D = start_paramter_res[3]
            prev_para_C = current_para_C
            prev_para_D = current_para_D
        current_meff = start_parameter_see[0]/degen
        current_gap = start_parameter_see[1]
        current_fermi_energy = start_parameter_see[2]
        prev_meff = current_meff
        prev_gap = current_gap
        prev_fermi_energy = current_fermi_energy
        
        parameter_res = start_paramter_res
        parameter_see = start_parameter_see
        
        # Save the number of iterations to track the progress
        number_iterations = 1
        
        # Infinite loop that ends if accuracy is reached or number of iterations is too large
        while True:
            if print_info:
                # print the current number of iterations
                print(f"Number of iteration: {number_iterations}")
                print("Starting Seebeck fit...")
                
            # Fit the Seebeck coefficient with the parameters 'mass', 'gap' and 'fermi_energy'
            if scatter_type == "acPh" or scatter_type == "dis":
                parameter_see, _ = opt.curve_fit(
                    lambda T, mass, gap, fermi_energy: te.dpb_see_vs_temp_single_scatter(temp_see_list, mass, gap, fermi_energy, degen, current_para_B, scal_fac = 1e6), 
                    temp_see_list, see_list * 1e6, p0 = start_parameter_see, bounds = limits_see, full_output=False, maxfev=1000)
            else:
                parameter_see, _ = opt.curve_fit(
                    lambda T, mass, gap, fermi_energy: te.dpb_see_vs_temp_double_scatter(temp_see_list, mass, gap, fermi_energy, degen, current_para_B, current_para_C, current_para_D, scal_fac = 1e6), 
                    temp_see_list, see_list * 1e6, p0 = start_parameter_see, bounds = limits_see, full_output=False, maxfev=1000)
            
            # Save the current band parameters
            current_meff = parameter_see[0]
            current_gap = parameter_see[1]
            current_fermi_energy = parameter_see[2]
            
            # Set the start parameters of the Sebeeck coefficient fit to the currently best values
            start_parameter_see = parameter_see
            
            if print_info:
                print(f"Mass: {current_meff:.5f}\n"
                      f"Gap: {current_gap:.5f}\n"
                      f"EF: {current_fermi_energy:.5f}\n")
                
            if print_info:
                print("Starting Resistivity fit...")
                     
            # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
            f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s = 0, T = temp_res_list, mass = current_meff, gap = current_gap, fermi_energy = current_fermi_energy, degen = degen)

            # Check if number of iterations is <= max_num_iterations to ensure that the loop is not infinite
            # If the fit is not converged after max_num_iterations steps, the resistivity is fitted one more time while keeping the factor for the Seebeck coefficient fixed
            if number_iterations > max_num_iterations:
                if print_info:
                    print(f"Error! Fitting not possible. Number of iterations in 'dpb_see_res_fit' exceeds {max_num_iterations}")                
                break

            # Fit the resistivity with the parameters 'para_A', para_B', 'para_C' and 'para_D'
            if scatter_type == "acPh":
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, meff: te.dpb_rho_vs_temp_acPh_new(T, para_A, meff, initial_weight = current_meff / current_para_B, gap = current_gap, fermi_energy = current_fermi_energy, degen = degen, scal_fac = 1e8),
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
                    # lambda T, para_A, para_B: te.dpb_rho_vs_temp_acPh(T, f0_1, f0_2, para_A, para_B, current_meff, scal_fac = 1e8), 
                    # temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            elif scatter_type == "dis":
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B: te.dpb_rho_vs_temp_dis(T, f0_1, f0_2, para_A, para_B, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            else:
                parameter_res, _ = opt.curve_fit(
                    lambda T, para_A, para_B, para_C, para_D: te.dpb_rho_vs_temp_acPh_dis(T, f0_1, f0_2, para_A, para_B, para_C, para_D, current_meff, scal_fac = 1e8), 
                    temp_res_list, res_list * 1e8, p0 = start_paramter_res, bounds = limits_res, full_output=False, maxfev=50_000)
            
            # Save the current scattering parameters
            current_para_A = parameter_res[0]
            current_para_B = parameter_res[1] / current_meff
            current_meff = parameter_res[1]
            print(f"Mass from res fit: {current_meff}")
            # current_para_B = parameter_res[1]
            if scatter_type == "acPhDis":
                current_para_C = parameter_res[2]
                current_para_D = parameter_res[3]
            
            # Set the start parameters of the resistivity fit to the currently best values
            start_paramter_res = parameter_res

            if print_info:
                # Print information about the fitting parameters of the resistivity
                if scatter_type == "acPh" or scatter_type == "dis":
                    print(f"current_para_A: {current_para_A:.5f}\n"
                        f"current_para_B: {current_para_B:.5f}\n")
                else:
                    print(f"current_para_A: {current_para_A:.5f}\n"
                        f"current_para_B: {current_para_B:.5f}\n"
                        f"current_para_C: {current_para_C:.5f}\n"
                        f"current_para_D: {current_para_D:.5f}")

            # Calculate the error of the fitting parameters of the resistivtiy
            error_meff         = np.abs((current_meff - prev_meff)/prev_meff)
            error_gap          = np.abs((current_gap - prev_gap)/prev_gap)
            error_fermi_energy = np.abs((current_fermi_energy - prev_fermi_energy)/prev_fermi_energy)
            error_para_A       = np.abs((current_para_A - prev_para_A)/prev_para_A)
            error_para_B       = np.abs((current_para_B - prev_para_B)/prev_para_B)
            if scatter_type == "acPhDis":
                error_para_C   = np.abs((current_para_C - prev_para_C)/prev_para_C)
                error_para_D   = np.abs((current_para_D - prev_para_D)/prev_para_D)
            
            # Check if the error of all fitting parameters is smaller than threshold
            if self.is_accuracy_reached(threshold, error_meff, error_gap, error_fermi_energy, error_para_A, error_para_B):
                if print_info:
                    print("Accuracy reached.")
                break
            if scatter_type == "acPhDis" and self.is_accuracy_reached(threshold, error_para_C, error_para_D):
                if print_info:
                    print("Accuracy reached.")
                break

            # Save the current fitting parameters to determine the error in the next step
            prev_meff = current_meff
            prev_gap = current_gap
            prev_fermi_energy = current_fermi_energy
            prev_para_A = current_para_A
            prev_para_B = current_para_B
            if scatter_type == "acPhDis":
                prev_para_C = current_para_C
                prev_para_D = current_para_D
            
            # Increase the number of iterations by 1
            number_iterations += 1
            
        if print_info:
            print("Starting Hall fit...")
            
        #Calculate Fermi integrals and rho at temperatures of hall
        f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s = 0, T = temp_hall_list, mass=current_meff, gap=current_gap, fermi_energy=current_fermi_energy, degen=degen)
        fm05_1, fm05_2 = te.dpb_fermiInt_vs_temp(s = -0.5, T = temp_hall_list, mass = current_meff, gap = current_gap, fermi_energy=current_fermi_energy, degen=degen)
        
        if scatter_type == "acPh":
            res_array_hall_temp = te.dpb_rho_vs_temp_acPh(temp_hall_list, f0_1, f0_2, current_para_A, current_para_B, current_meff)
        if scatter_type == "dis":
            res_array_hall_temp = te.dpb_rho_vs_temp_dis(temp_hall_list, f0_1, f0_2, current_para_A, current_para_B, current_meff)
        if scatter_type == "acPhDis":
            res_array_hall_temp = te.dpb_rho_vs_temp_acPh_dis(temp_hall_list, f0_1, f0_2, current_para_A, current_para_B, current_para_C, current_para_D, current_meff)
        
        # Fit the hall with the parameter 'para_E'
        if scatter_type == "acPh":
            parameter_hall, _ = opt.curve_fit(
            lambda T, para_E: te.dpb_hall_vs_temp_acPh(T, res_array_hall_temp, fm05_1, fm05_2, current_meff, degen, current_para_A, current_para_B, para_E, scal_fac = 1e10), 
            temp_hall_list, hall_list * 1e10, p0 = initial_para_E, bounds = limits_para_E, full_output=False, maxfev=5000)
        elif scatter_type == "dis":
            parameter_hall, _ = opt.curve_fit(
            lambda T, para_E: te.dpb_hall_vs_temp_dis(T, res_array_hall_temp, fm05_1, fm05_2, current_meff, degen, current_para_A, current_para_B, para_E, scal_fac = 1e10), 
            temp_hall_list, hall_list * 1e10, p0 = initial_para_E, bounds = limits_para_E, full_output=False, maxfev=5000)
        else:
            parameter_hall, _ = opt.curve_fit(
                lambda T, para_E: te.dpb_hall_vs_temp_acPh_dis(T, res_array_hall_temp, fm05_1, fm05_2, current_meff, degen, current_para_A, current_para_B, current_para_C, current_para_D, para_E, scal_fac = 1e10), 
                temp_hall_list, hall_list * 1e10, p0 = initial_para_E, bounds = limits_para_E, full_output=False, maxfev=5000)

        # Save the current scattering parameters
        current_para_E = parameter_hall[0]
        current_mass1 = current_para_E**(2/3)
        
        # Print the number of steps needed to reach the desired accuracy
        if print_info:
            print(f"\nIt took {number_iterations} iterations to evaluate the best fit\n")
        
        if print_info:           
            print("###########################################")
            print("Band structure parameters:")
            print(
                f"Mass: {current_meff:.5f}\n"
                f"Gap: {current_gap:.5f}\n"
                f"EF: {current_fermi_energy:.5f}\n"
                )
            print("Scattering parameters:")
            if scatter_type == "acPh" or scatter_type == "dis":
                print(
                    f"parameter A: {current_para_A:.3f}\n"
                    f"parameter B: {current_para_B:.3f}\n"
                    )
            else:
                print(
                    f"parameter A: {current_para_A:.3f}\n"
                    f"parameter B: {current_para_B:.3f}\n"
                    f"parameter C: {current_para_C:.3f}\n"
                    f"parameter D: {current_para_D:.3f}\n"
                    )
            print(
                f"parameter E: {current_para_E}\n"
                f"band mass 1: {current_mass1}\n"
            )
            print("###########################################")
        
        print("Finished 'dpb_see_res_hall_fit'")
        
        # Save all fitting parameters in a list and return the list
        parameters[0:3] = [current_meff*degen, current_gap, current_fermi_energy]
        parameters[9] = current_mass1
        if scatter_type == "acPh":
            parameters[3:5] = [current_para_A, current_para_B]
        elif scatter_type == "dis":
            parameters[7:9] = [current_para_A, current_para_B]
        else:
            parameters[3:7] = [current_para_A, current_para_B, current_para_C/300., current_para_D/300.] #adjusted, as the input is given at 300 K
            
        # if scatter_type == "acPh" or scatter_type == "dis":
        #     parameters = [current_meff, current_gap, current_fermi_energy, current_para_A, current_para_B, current_mass1]
        # else:
        #     parameters = [current_meff, current_gap, current_fermi_energy, current_para_A, current_para_B, current_para_C, current_para_D, current_mass1]
        return parameters
    
    
    '''
    #################################################
    Functions to calculate temperature-dependent thermoelectric properties based on the model and mode
    #################################################
    '''
    
    # Calculates the temperature-dependent Seebeck coefficient of a single band from given band-structure parameters
    def spb_see_calc(
        self, 
        Tmin: float, 
        Tmax: float, 
        Tnumb: int, 
        band_mass: float, 
        fermi_energy: float
        ) -> tuple[np.array, np.array]:
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        see_array = te.spb_see_vs_temp(temp_array, fermi_energy, band_mass, scal_fac = 1)
        
        return temp_array, see_array
    
    # Calculates the temperature-dependent Seebeck coefficient two bands from given band-structure parameters
    def dpb_see_calc(
        self, 
        Tmin: float, 
        Tmax: float, 
        Tnumb: int, 
        parameter_see: list[float], 
        degen: float
        ) -> tuple[np.array, np.array]:
        
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        see_array = te.dpb_see_vs_temp(temp_array, meff, gap, fermi_energy, degen, scal_fac = 1)
        
        return temp_array, see_array
    
    # Calculates the temperature-dependent Seebeck coefficient two bands from given band-structure and scattering parameters
    def dpb_see_with_scatter_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        parameter_res: list[float],
        degen: float,
        scatter_type: str
        ) -> tuple[np.array, np.array, np.array]:
        
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_see_res_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[2:6]
        elif scatter_type == "dis":
            del parameter_res[0:4]
        else:
            del parameter_res[4:6]
            for i in range(2,4):
                parameter_res[i] = parameter_res[i]*300. #adjusted, as the input is given at 300 K

        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the scattering parameters
        para_B = parameter_res[1]
        if scatter_type == "acPhDis":
            para_C = parameter_res[2]
            para_D = parameter_res[3]
            
        # Define the band structure parameters
        mass = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]


        # Calculate the Seebeck coefficient
        if scatter_type == "acPh" or scatter_type == "dis":
            see_array = te.dpb_see_vs_temp_single_scatter(temp_array, mass, gap, fermi_energy, degen, para_B, scal_fac = 1)
        else:
            see_array = te.dpb_see_vs_temp_double_scatter(temp_array, mass, gap, fermi_energy, degen, para_B, para_C, para_D, scal_fac = 1)
    
        return temp_array, see_array
    
    # Calculates the temperature-dependent Seebeck coefficient and electrical resistivity of a single band from given band-structure and scattering parameters
    def spb_see_res_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        band_mass: float, 
        fermi_energy: float,
        parameter_res: list[float],
        scatter_type: str
        ) -> list[float]:
        
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'spb_see_res_calc' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[1:3]
        elif scatter_type == "dis":
            del parameter_res[0:2]
        else:
            del parameter_res[-1]
            parameter_res[1] = parameter_res[1]*300. #adjusted, as the input is given at 300 K

        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        if scatter_type == "acPhDis":
            para_B = parameter_res[1]
        
        # Calculate the Seebeck coefficient
        see_array = te.spb_see_vs_temp(temp_array, fermi_energy, band_mass, scal_fac = 1)
                     
        # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0 = te.spb_fermiInt_vs_temp(s = 0, T = temp_array, mass_sign = np.sign(band_mass), fermi_energy = fermi_energy)

        # Calculate the resistivity
        if scatter_type == "acPh":
            res_array = te.spb_rho_vs_temp_acPh(temp_array, f0, para_A, scal_fac = 1)

        elif scatter_type == "dis":
            res_array = te.spb_rho_vs_temp_dis(temp_array, f0, para_A, scal_fac = 1)
        else:
            res_array = te.spb_rho_vs_temp_acPh_dis(temp_array, f0, para_A, para_B, scal_fac = 1)
            
        return temp_array, see_array, res_array
    
    # Calculates the temperature-dependent Seebeck coefficient and electircal resistivity of two bands from given band-structure and scattering parameters
    def dpb_see_res_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        parameter_res: list[float],
        degen: float,
        scatter_type: str
        ) -> tuple[np.array, np.array, np.array]:
        """Fits the Seebeck coefficient and resistivity data within the double-parabolic-band model and returns band-structure and scattering parameters
        
        Parameters
        ----------
        Tmin : float
            Lowest temperature at which the thermoelectric properties are calcualted
        Tmax : float
            Highest temperature at which the thermoelectric properties are calcualted
        Tnumb : int
            Number of temperatures at which the thermoelectric properties are calculated
        initial_parameter_see : list[float]
            Initial fitting parameters of the Sebeeck coefficient
        initial_parameter_res : list[float]
            Initial fitting parameters of the resistivity
        scatter_type : string
            Determines the predominant scattering mechanisms, must be either 'acPh' (acoustic-phonon scattering), 'dis' (alloy-disorder scattering) or 'acPhDis' (both)
        degen : float
            Ratio of the degeneracy of band 2 to band 1       
            
        Returns
        -------
        out : tuple[np.array, np.array, np.array]
        """
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_see_res_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[2:6]
        elif scatter_type == "dis":
            del parameter_res[0:4]
        else:
            del parameter_res[4:6]
            for i in range(2,4):
                parameter_res[i] = parameter_res[i]*300. #adjusted, as the input is given at 300 K

        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        para_B = parameter_res[1]
        if scatter_type == "acPhDis":
            para_C = parameter_res[2]
            para_D = parameter_res[3]
            #print(para_C, para_D)
        # Define the band structure parameters
        mass = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]


        # Calculate the Seebeck coefficient
        if scatter_type == "acPh" or scatter_type == "dis":
            see_array = te.dpb_see_vs_temp_single_scatter(temp_array, mass, gap, fermi_energy, degen, para_B, scal_fac = 1)
        else:
            see_array = te.dpb_see_vs_temp_double_scatter(temp_array, mass, gap, fermi_energy, degen, para_B, para_C, para_D, scal_fac = 1)
        

        # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s= 0 ,T = temp_array, mass = mass, gap = gap, fermi_energy = fermi_energy, degen = degen)

        # Calculate the electrical resistivity
        if scatter_type == "acPh":
            res_array =  te.dpb_rho_vs_temp_acPh(temp_array, f0_1, f0_2, para_A, para_B, mass, scal_fac = 1)
        elif scatter_type == "dis":
            res_array = te.dpb_rho_vs_temp_dis(temp_array, f0_1, f0_2, para_A, para_B, mass, scal_fac = 1)
        else: 
            res_array = te.dpb_rho_vs_temp_acPh_dis(temp_array, f0_1, f0_2, para_A, para_B, para_C, para_D, mass, scal_fac = 1)

        return temp_array, see_array, res_array
    
    # Calculates the temperature-dependent Seebeck coefficient, electrical resistivity and Hall coefficient of a single band from given band-structure and scattering parameters
    def spb_see_res_hall_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        band_mass: float, 
        fermi_energy: float,
        parameter_res: list[float],
        scatter_type: str
        ) -> list[float]:
        
        # Check if scatter_type is properly set to acPh, dis or acPhAlDi
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'spb_see_res_calc' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[1:3]
        elif scatter_type == "dis":
            del parameter_res[0:2]
        else:
            del parameter_res[-1]
            parameter_res[1] = parameter_res[1]*300. #adjusted, as the input is given at 300 K
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        para_E = np.abs(band_mass)**(3/2)*-np.sign(band_mass) #!!! Hotfixed this has to be looked at!
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        if scatter_type == "acPhDis":
            para_B = parameter_res[1]
        
        # Calculate the Seebeck coefficient
        see_array = te.spb_see_vs_temp(temp_array, fermi_energy, band_mass, scal_fac = 1)
                     
        # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0 = te.spb_fermiInt_vs_temp(s = 0, T = temp_array, mass_sign = np.sign(band_mass), fermi_energy = fermi_energy)

        # Calculate the resistivity
        if scatter_type == "acPh":
            res_array = te.spb_rho_vs_temp_acPh(temp_array, f0, para_A, scal_fac = 1)

        elif scatter_type == "dis":
            res_array = te.spb_rho_vs_temp_dis(temp_array, f0, para_A, scal_fac = 1)
        else:
            res_array = te.spb_rho_vs_temp_acPh_dis(temp_array, f0, para_A, para_B, scal_fac = 1)
        
        # Calculate the Hall coefficient
        hall_array = te.spb_hall_vs_temp(temp_array, fermi_energy, np.sign(band_mass), para_E)

        return temp_array, see_array, res_array, hall_array

    # Calculates the temperature-dependent Seebeck coefficient, electrical resistivity and Hall coefficient of two bands from given band-structure and scattering parameters
    def dpb_see_res_hall_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        parameter_res: list[float],
        band_mass1: float,
        degen: float,
        scatter_type: str
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits the Seebeck coefficient and resistivity data within the double-parabolic-band model and returns band-structure and scattering parameters
    
        Parameters
        ----------
        Tmin : float
            Lowest temperature at which the thermoelectric properties are calcualted
        Tmax : float
            Highest temperature at which the thermoelectric properties are calcualted
        Tnumb : int
            Number of temperatures at which the thermoelectric properties are calculated
        initial_parameter_see : list[float]
            Initial fitting parameters of the Sebeeck coefficient
        initial_parameter_res : list[float]
            Initial fitting parameters of the resistivity
        scatter_type : string
            Determines the predominant scattering mechanisms, must be either 'acPh' (acoustic-phonon scattering), 'dis' (alloy-disorder scattering) or 'acPhDis' (both)
        degen : float
            Ratio of the degeneracy of band 2 to band 1       
            
        Returns
        -------
        out : tuple[np.array, np.array, np.array]
        """
        # Check if scatter_type is properly set to acPh, alDi or acPhAlDi
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_see_res_fit' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[2:6]
        elif scatter_type == "dis":
            del parameter_res[0:4]
        else:
            del parameter_res[4:6]
            for i in range(2,4):
                parameter_res[i] = parameter_res[i]*300. #adjusted, as the input is given at 300 K
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        para_E = [band_mass1**(3/2)]
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        para_B = parameter_res[1]
        if scatter_type == "acPhDis":
            para_C = parameter_res[2]
            para_D = parameter_res[3]
            
        # Define the band structure parameters
        mass = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]


        # Calculate the Seebeck coefficient
        if scatter_type == "acPh" or scatter_type == "dis":
            see_array = te.dpb_see_vs_temp_single_scatter(temp_array, mass, gap, fermi_energy, degen, para_B, scal_fac = 1)
        else:
            see_array = te.dpb_see_vs_temp_double_scatter(temp_array, mass, gap, fermi_energy, degen, para_B, para_C, para_D, scal_fac = 1)

        # Calculate Fermi integrals
        f0_1, f0_2 = te.dpb_fermiInt_vs_temp( s= 0, T = temp_array, mass = mass, gap = gap, fermi_energy = fermi_energy, degen = degen)
        fm05_1, fm05_2 = te.dpb_fermiInt_vs_temp(s = -0.5, T = temp_array, mass = mass, gap = gap, fermi_energy = fermi_energy, degen = degen)

        # Calculate the electrical resistivity
        if scatter_type == "acPh":
            res_array =  te.dpb_rho_vs_temp_acPh(temp_array, f0_1, f0_2, para_A, para_B, mass, scal_fac = 1)
        elif scatter_type == "dis":
            res_array = te.dpb_rho_vs_temp_dis(temp_array, f0_1, f0_2, para_A, para_B, mass, scal_fac = 1)
        else:
            res_array = te.dpb_rho_vs_temp_acPh_dis(temp_array, f0_1, f0_2, para_A, para_B, para_C, para_D, mass, scal_fac = 1)

        # Calculate the Hall coefficient
        if scatter_type == "acPh":
            hall_array = te.dpb_hall_vs_temp_acPh(temp_array, res_array, fm05_1, fm05_2, mass, degen, para_A, para_B, para_E, scal_fac = 1)
        elif scatter_type == "dis":
            hall_array = te.dpb_hall_vs_temp_dis(temp_array, res_array, fm05_1, fm05_2, mass, degen, para_A, para_B, para_E, scal_fac = 1)
        else:
            hall_array = te.dpb_hall_vs_temp_acPh_dis(temp_array, res_array, fm05_1, fm05_2, mass, degen, para_A, para_B, para_C, para_D, para_E, scal_fac = 1)

        return temp_array, see_array, res_array, hall_array
    
    
    '''
    #################################################
    Functions to calculate the functions shown as additional information
    #################################################
    '''

    '''
    SPB
    '''
 
    def spb_elecCond_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        band_mass: float, 
        fermi_energy: float,
        parameter_res: list[float],
        scatter_type: str
        ):
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[1:3]
        elif scatter_type == "dis":
            del parameter_res[0:2]
        else:
            del parameter_res[-1]

        # Define the scattering parameters
        para_A = parameter_res[0]
        if scatter_type == "acPhDis":
            para_B = parameter_res[1]
                     
        # Calculate the 'pristine' conductivity of the band from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0 = te.spb_fermiInt_vs_temp(s = 0, T = temp_array, mass_sign = np.sign(band_mass), fermi_energy = fermi_energy)
        
        # Calculate the resistivity
        if scatter_type == "acPh":
            return temp_array, te.spb_elecCond_vs_temp_acPh(temp_array, f0, para_A, scal_fac = 1)

        elif scatter_type == "dis":
            return temp_array, te.spb_elecCond_vs_temp_dis(temp_array, f0, para_A, scal_fac = 1)
        else:
            return temp_array, te.spb_elecCond_vs_temp_acPh_dis(temp_array, f0, para_A, para_B, scal_fac = 1)
    
    def spb_chemPot_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        band_mass: int,
        fermi_energy: float
        ):
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        return temp_array, te.spb_chemPot_vs_temp(temp_array, band_mass, fermi_energy)
    
    def spb_thermCond_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        band_mass: float, 
        fermi_energy: float,
        parameter_res: list[float],
        scatter_type: str
        ):
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[1:3]
        elif scatter_type == "dis":
            del parameter_res[0:2]
        else:
            del parameter_res[-1]

        # Define the scattering parameters
        para_A = parameter_res[0]
        if scatter_type == "acPhDis":
            para_B = parameter_res[1]
                     
        # Calculate the 'pristine' conductivity of the band from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0 = te.spb_fermiInt_vs_temp(s = 0, T = temp_array, mass_sign = np.sign(band_mass), fermi_energy = fermi_energy)
        
        # Calculate the resistivity
        if scatter_type == "acPh":
            return temp_array, te.spb_thermCond_vs_temp_acPh(temp_array, f0, fermi_energy, band_mass, para_A, scal_fac = 1)

        elif scatter_type == "dis":
            return temp_array, te.spb_thermCond_vs_temp_dis(temp_array, f0, fermi_energy, band_mass, para_A, scal_fac = 1)
        else:
            return temp_array, te.spb_thermCond_vs_temp_acPh_dis(temp_array, f0, fermi_energy, band_mass, para_A, para_B, scal_fac = 1)
    
    '''
    DPB
    '''
    
    def dpb_ind_see_seeOnly_calc(
        self, 
        Tmin: float, 
        Tmax: float, 
        Tnumb: int, 
        parameter_see: list[float], 
        degen: float
    ):
        
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        ind_see1, ind_see2 = te.dpb_indCont_see_vs_temp(temp_array, meff, gap, fermi_energy, degen, scal_fac = 1)
        
        return temp_array, ind_see1, ind_see2
    
    def dpb_ind_see_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        parameter_res: list[float],
        degen: float,
        scatter_type: str
        ) -> tuple[np.array, np.array, np.array]:
            
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_ind_see_calc' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[2:6]
        elif scatter_type == "dis":
            del parameter_res[0:4]
        else:
            del parameter_res[4:6]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        para_B = parameter_res[1]
        if scatter_type == "acPhDis":
            para_C = parameter_res[2]
            para_D = parameter_res[3]
            
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]

        # Calculate the Seebeck coefficient
        if scatter_type == "acPh" or scatter_type == "dis":
            ind_see1, ind_see2 = te.dpb_indCont_see_vs_temp_single_scatter(temp_array, meff, gap, fermi_energy, degen, para_B, scal_fac = 1)
        else:
            ind_see1, ind_see2 = te.dpb_indCont_see_vs_temp_double_scatter(temp_array, meff, gap, fermi_energy, degen, para_B, para_C, para_D, scal_fac = 1)

        return temp_array, ind_see1, ind_see2
    
    def dpb_chemPot_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        degen: float
        ):
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        chemPot_array = te.dpb_chemPot_vs_temp(temp_array, gap, meff, fermi_energy, degen)
        
        return temp_array, chemPot_array
    
    def dpb_ind_elecCond_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        parameter_res: list[float],
        degen: float,
        scatter_type: str
    ):
        
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_ind_elecCond_calc' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[2:6]
        elif scatter_type == "dis":
            del parameter_res[0:4]
        else:
            del parameter_res[4:6]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        para_B = parameter_res[1]
        if scatter_type == "acPhDis":
            para_C = parameter_res[2]
            para_D = parameter_res[3]
            
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s= 0 ,T = temp_array, mass = meff, gap = gap, fermi_energy = fermi_energy, degen = degen)

        # Calculate the electrical resistivity
        if scatter_type == "acPh":
            sig1, sig2 = te.dpb_ind_elecCond_vs_temp_acPh(temp_array, f0_1, f0_2, para_A, para_B, meff, scal_fac = 1)
        elif scatter_type == "dis":
            sig1, sig2 = te.dpb_ind_elecCond_vs_temp_dis(temp_array, f0_1, f0_2, para_A, para_B, meff, scal_fac = 1)
        else:
            sig1, sig2 = te.dpb_ind_elecCond_vs_temp_acPh_dis(temp_array, f0_1, f0_2, para_A, para_B, para_C, para_D, meff, scal_fac = 1)
            
        return temp_array, sig1, sig2
    
    def dpb_ind_carCon_calc(
        self, 
        Tmin: float, 
        Tmax: float, 
        Tnumb: int, 
        parameter_see: list[float], 
        degen: float
    ):
        
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        ind_charCon1, ind_charCon2 = te.dpb_ind_carCon_vs_temp(temp_array, fermi_energy, meff, gap, degen)
        
        return temp_array, ind_charCon1, ind_charCon2
    
    def dpb_ind_carCon_calc_m1(
        self, 
        Tmin: float, 
        Tmax: float, 
        Tnumb: int, 
        parameter_see: list[float], 
        degen: float,
        m1: float
    ):
        
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        ind_charCon1, ind_charCon2 = te.dpb_ind_carCon_vs_temp(temp_array, fermi_energy, meff, gap, degen)
        
        return temp_array, m1**1.5 * ind_charCon1, m1**1.5 * ind_charCon2
    
    def dpb_elecCond_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        parameter_res: list[float],
        degen: float,
        scatter_type: str
    ):
        
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_thermCond_calc' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[2:6]
        elif scatter_type == "dis":
            del parameter_res[0:4]
        else:
            del parameter_res[4:6]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        para_B = parameter_res[1]
        if scatter_type == "acPhDis":
            para_C = parameter_res[2]
            para_D = parameter_res[3]
            
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s= 0 ,T = temp_array, mass = meff, gap = gap, fermi_energy = fermi_energy, degen = degen)

        # Calculate the electrical resistivity
        if scatter_type == "acPh":
            cond = 1 / te.dpb_rho_vs_temp_acPh(temp_array, f0_1, f0_2, para_A, para_B, meff, scal_fac = 1)
        elif scatter_type == "dis":
            cond = 1 / te.dpb_rho_vs_temp_dis(temp_array, f0_1, f0_2, para_A, para_B, meff, scal_fac = 1)
        else:
            cond = 1 / te.dpb_rho_vs_temp_acPh_dis(temp_array, f0_1, f0_2, para_A, para_B, para_C, para_D, meff, scal_fac = 1)
            
        return temp_array, cond
    
    def dpb_thermCond_calc(
        self,
        Tmin: float, 
        Tmax: float, 
        Tnumb: int,
        parameter_see: list[float],
        parameter_res: list[float],
        degen: float,
        scatter_type: str
    ):
        
        # Check if scatter_type is properly set to acPh, dis or acPhDis
        if scatter_type != "acPh" and scatter_type != "dis" and scatter_type != "acPhDis":
            raise Exception("Error! Parameter 'scatter_type' in function 'dpb_thermCond_calc' must either be 'acPh', 'dis' or 'acPhDis'")
        
        # Change parameter arrays for the fit according to the scattering type:
        if scatter_type == "acPh":
            del parameter_res[2:6]
        elif scatter_type == "dis":
            del parameter_res[0:4]
        else:
            del parameter_res[4:6]
        
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        
        # Define the scattering parameters
        para_A = parameter_res[0]
        para_B = parameter_res[1]
        if scatter_type == "acPhDis":
            para_C = parameter_res[2]
            para_D = parameter_res[3]
            
        # Define the band structure parameters
        meff = parameter_see[0]/degen
        gap = parameter_see[1]
        fermi_energy = parameter_see[2]
        
        # Calculate the 'pristine' conductivities of band 1 and band 2 from the fitted band-structure parameters with K = 1, abs(meff) = 1 and beta = 0
        f0_1, f0_2 = te.dpb_fermiInt_vs_temp(s= 0 ,T = temp_array, mass = meff, gap = gap, fermi_energy = fermi_energy, degen = degen)

        # Calculate the electrical resistivity
        if scatter_type == "acPh":
            thermCond = te.dpb_thermCond_vs_temp_acPh(temp_array, f0_1, f0_2, para_A, para_B, meff, gap, fermi_energy, degen, scal_fac = 1)
        elif scatter_type == "dis":
            thermCond = te.dpb_thermCond_vs_temp_dis(temp_array, f0_1, f0_2, para_A, para_B, meff, gap, fermi_energy, degen, scal_fac = 1)
        else:
            thermCond = te.dpb_thermCond_vs_temp_acPh_dis(temp_array, f0_1, f0_2, para_A, para_B, para_C, para_D, meff, gap, fermi_energy, degen, scal_fac = 1)
    
        return temp_array, thermCond
    
    '''
    #################################################
    Temporary helper functions
    #################################################
    '''
    
    # Calculates the parabolic band structure
    def get_parabolic_band(self, kmin, kmax, knumb, mass, energy):
        k_array = np.linspace(kmin, kmax, knumb)
        energy_array = np.zeros(knumb)
        for i, wavevector in enumerate(k_array):
            energy_array[i] = te.parabolic_band(wavevector, mass, energy)
        return k_array, energy_array
    
    def ind_seebeck_2PB(self, Tmin, Tmax, Tnumb, mass, gap, fermi_energy, degen):
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        see1_array = np.zeros(Tnumb)
        see2_array = np.zeros(Tnumb)
        for i, temp in enumerate(temp_array):
            see1_array[i], see2_array[i] = te.spart_2PB(temp, mass, gap, degen, fermi_energy)
            
        return temp_array, see1_array, see2_array
    
    def ind_cc_2PB(self, Tmin, Tmax, Tnumb, mass, gap, fermi_energy, degen):
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        chargeCon1_array = np.zeros(Tnumb)
        chargeCon2_array = np.zeros(Tnumb)
        
        for i, temp in enumerate(temp_array):
            chargeCon1_array[i], chargeCon2_array[i] = te.dpb_ind_carCon(temp, fermi_energy, mass, gap, degen)

        return temp_array, chargeCon1_array, chargeCon2_array

    #!!! FILLER FUNCTION, NOT FUNCTIONAL AT ALL
    def chempot_1PB(self,Tmin,Tmax,Tnumb,mass, fermi_energy, degen):
        temp_array = [1,2,3]
        chemPot_array = [1,2,3]
        # eta = fermi_energy
        
        # for i, temp in enumerate(temp_array):
        #     eta = te.mu2PB_solve(eta, temp, gap, mass, fermi_energy, degen)
        #     chemPot_array[i] = eta

        return temp_array, chemPot_array
        
    
    def chempot_2PB(self,Tmin,Tmax,Tnumb,mass,gap,fermi_energy, degen):
        temp_array = np.linspace(Tmin, Tmax, Tnumb)
        chemPot_array = np.zeros(Tnumb)
        eta = fermi_energy
        
        for i, temp in enumerate(temp_array):
            eta = te.dpb_chemPot_solve(eta, temp, gap, mass, fermi_energy, degen)
            chemPot_array[i] = eta

        return temp_array, chemPot_array
    
if __name__ == "__main__":
    initial_fermi_energy = [300]
    initial_parameter_res = [1, 2, 3]
    parameters = initial_fermi_energy + initial_parameter_res
    
    del initial_parameter_res[1:3]
    
    #print(parameters)
    
    parameters2 = [0, 1, 2, 3, 4, 5]
    parameters2[1:3] = [0, 9]
    #print(parameters2)