import os
import glob
import hashlib
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import argrelmin
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from scipy.ndimage import label
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


#--------------------------------#
#Define functions and classes.
#--------------------------------#

def load_sample_subset(filepath):
    """ 
    Load a stellar sample from a CSV file and extract relevant columns.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the stellar sample.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing only the selected columns.
    
    """
        
    # Define the columns to extract
    columns = [
        'primary_name', 'Mass', 'Teff(K)',
        'Mass-err', 'Mass+err', 'Teff_err(K)',
        'logLuminosity', 'logLuminosity-err', 'logLuminosity+err'
    ]

    # Load the full sample and return only the desired columns
    sample = pd.read_csv(filepath)
    return sample[columns]

#--------------------------------#

def time_series_properties(nu_grid):
    """
    Given a frequency grid in μHz, compute the time series length and cadence in seconds.

    Parameters
    ----------
    nu_grid : ndarray
        Frequency grid in microhertz (μHz).

    Returns
    -------
    time_length_sec : float
        Total duration of the time series in seconds.
        
    cadence_sec : float
        Cadence (sampling interval) of the time series in seconds.
    
    """
    
    if len(nu_grid) < 2:
        raise ValueError("Frequency grid must contain at least two points.")

    # Frequency resolution in Hz (μHz to Hz)
    delta_nu_hz = (nu_grid[1] - nu_grid[0]) * 1e-6
    time_length_sec = 1 / delta_nu_hz

    # Nyquist frequency in Hz
    nu_max_hz = np.max(nu_grid) * 1e-6
    cadence_sec = 1 / (2 * nu_max_hz)

    return time_length_sec, cadence_sec


#--------------------------------#

def sample_truncated_gaussian(mean, err_minus, err_plus, N):
    """
    Draw N samples from a symmetric Gaussian centered on `mean` with sigma
    derived from asymmetric errors, and impose a strict physical lower bound.
    
    Parameters
    ----------
    mean : float
        Measured value (central estimate).
    err_minus : float
        Negative uncertainty (lower error).
    err_plus : float
        Positive uncertainty (upper error).
    N : int
        Number of samples to draw.
    
    Returns
    -------
    samples : ndarray
        Array of N sampled values.
    """
    
    # Convert asymmetric errors to a single sigma
    sigma = 0.5 * (err_minus + err_plus)
    if sigma == 0:
        return np.full(N, mean, dtype=float)
    
    # Draw normal samples from a gaussian distribution
    samples = np.random.normal(loc=mean, scale=sigma, size=N)
    # Enforce lower physical limit
    samples = np.maximum(samples, 1e-3)
    
   
    return samples

#----------------------------------------------------#

def numax(M, R, T_eff):
    """
    Calculate ν_max (frequency of maximum oscillation power) for a star.

    Parameters
    ----------
    M : float or array
        Stellar mass in solar masses.
    R : float or array
        Stellar radius in solar radii.
    T_eff : float or array
        Effective temperature in Kelvin.

    Returns
    -------
    float or array
        ν_max in µHz.
    """
    numax_solar = 3050.0     # Solar ν_max in µHz
    T_eff_solar = 5777.0     # Solar effective temperature in K
    return M * R**(-2) * (T_eff / T_eff_solar)**(-0.5) * numax_solar

#----------------------------------------------------#

def dnu(M, R):
    """
    Calculate Δν (large frequency separation) for a star.

    Parameters
    ----------
    M : float or array
        Stellar mass in solar masses.
    R : float or array
        Stellar radius in solar radii.

    Returns
    -------
    float or array
        Δν in µHz.
    """
    dnu_solar = 135.0   # Solar Δν in µHz
    return M**0.5 / R**1.5 * dnu_solar

#---------------------------------------------------#

def radius(T_eff, L):
    """
    Compute stellar radius from luminosity and effective temperature.

    Parameters
    ----------
    T_eff : float or array
        Effective temperature in Kelvin.
    L : float or array
        Stellar luminosity in solar units.

    Returns
    -------
    float or array
        Stellar radius in solar radii.
    """
    T_eff_solar = 5777.0
    return (L**0.5) * (T_eff / T_eff_solar)**(-2)

#--------------------------------#

def monte_carlo(index, sample_file, N, base_path):
    
    """
    Perform a Monte Carlo sampling to generate N realisations of stellar parameters
    for a given star, based on its uncertainties. 

    Parameters
    ----------
    index : int
        Row index of the star in the sample CSV.
    sample_file : str
        Path to the CSV containing stellar parameters and uncertainties.
    N : int
        Number of Monte Carlo realisations to retain after filtering.
    base_path : str
        Base directory where the star's folder and output CSV will be saved.
    """

    # ------------------------------------------------------------
    # 1. Load the sample dataset and select the star of interest
    # ------------------------------------------------------------
    sample = pd.read_csv(sample_file)
    row = sample.loc[index]
    ID = row['primary_name']

    # ------------------------------------------------------------
    # 2. Extract central values and uncertainties
    # ------------------------------------------------------------
    M0 = row['Mass']
    T0 = row['Teff(K)']
    L0 = 10 ** row['logLuminosity']  # Convert log L to linear L

    # Uncertainties (for truncated normal)
    M_punc = abs(row['Mass+err'])
    M_nunc = abs(row['Mass-err'])
    T_unc = abs(row['Teff_err(K)'])
    # Compute upper/lower luminosities
    L_punc = 10 ** row['logLuminosity+err']
    L_nunc = 10 ** row['logLuminosity-err']

    # ------------------------------------------------------------
    # 3. Oversample the input distributions
    # ------------------------------------------------------------
    oversample_factor = 5
    sample_size = oversample_factor * N

    # Draw many more samples than needed to enable filtering later
    M_all = sample_truncated_gaussian(M0, M_nunc, M_punc, sample_size)
    T_all = sample_truncated_gaussian(T0, T_unc, T_unc, sample_size)
    L_all = sample_truncated_gaussian(L0, L_nunc, L_punc, sample_size)

    # ------------------------------------------------------------
    # 4. Compute derived stellar quantities for each sample
    # ------------------------------------------------------------
    R_all = radius(T_all, L_all)            # Radius from L and Teff
    numax_all = numax(M_all, R_all, T_all)  # νmax scaling relation
    dnu_all = dnu(M_all, R_all)             # Δν density scaling

    # ------------------------------------------------------------
    # 5. Apply filtering criteria
    # ------------------------------------------------------------
    # Exclude unphysically high νmax realisations
    mask = numax_all <= 9000

    # Ensure we have at least N valid realisations
    valid_count = np.sum(mask)
    if valid_count < N:
        raise ValueError(
            f"Only {valid_count} valid samples (numax <= 9000). "
            f"Expected N={N}. Increase oversampling or inspect inputs."
        )

    # ------------------------------------------------------------
    # 6. Take the first N valid samples
    # ------------------------------------------------------------
    M_samples = M_all[mask][:N]
    T_samples = T_all[mask][:N]
    L_samples = L_all[mask][:N]
    R_samples = R_all[mask][:N]
    numax_samples = numax_all[mask][:N]
    dnu_samples = dnu_all[mask][:N]

    # ------------------------------------------------------------
    # 7. Save MC parameter table to disk for reproducibility
    # ------------------------------------------------------------
    params_df = pd.DataFrame({
        "realisation": np.arange(1, N + 1),
        "M": M_samples,
        "R": R_samples,
        "Teff": T_samples,
        "L": L_samples,
        "numax": numax_samples,
        "dnu": dnu_samples
    })

    # Create folder named after the star
    main_folder_ID = str(ID).replace("/", "_").replace(" ", "_")
    main_folder = os.path.join(base_path, main_folder_ID)
    os.makedirs(main_folder, exist_ok=True)

    csv_path = os.path.join(main_folder, f"{main_folder_ID}_MC_parameters.csv")
    params_df.to_csv(csv_path, index=False)

#--------------------------------#

class model:
    """
    Stellar oscillation model for solar-like stars.
    Computes mode frequencies, linewidths, and powers for l=0,1,2.
    """

    def __init__(self, nu):
        """Initialize with frequency grid and basic mode settings."""
        self.nu = nu
        self.N_p = np.arange(0, 40, 1)            # Radial orders l=0
        self.vis = {'V10': 1.35, 'V20': 1.02}  # Visibility factors for l=1,2 modes from HARPS data (Handberg & Lund 2011: https://www.aanda.org/articles/aa/abs/2011/03/aa15451-10/aa15451-10.html)

    # -----------------------------
    # Mode frequency calculations
    # -----------------------------
    def asymptotic_nu_p(self, numax, dnu):
        """
        Compute l=0 mode frequencies from asymptotic relation (μHz) (Nielson et al. 2011: https://iopscience.iop.org/article/10.3847/1538-3881/abcd39/meta ).
        
        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
            
        Returns
        -------
        nu0s : ndarray
            Array of l=0 mode frequencies from the asymptotic relation (muHz).
        """
        
        eps_p = 1.48   # Phase term for main-sequence stars. Solar value ~1.48 +/- 0.02 (White et al. 2011: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1109.3455)   
        a_MS = 0.57   #~0.57 +/- 0.02 (Mosser et al. 2013: https://www.aanda.org/articles/aa/abs/2013/02/aa20435-12/aa20435-12.html )
        n_max = numax / dnu - eps_p
        alpha_p = 2 * a_MS * n_max**(-2) #Curvature term for main-sequence stars.
        n_p_max = numax / dnu
        return (self.N_p + eps_p + alpha_p/2*(self.N_p - n_p_max)**2) * dnu

    # -----------------------------
    # Lorentzian profile
    # -----------------------------
    def lor(self, nu0, h, w):
        """
        Lorentzian to describe an oscillation mode.

        Parameters
        ----------
        nu0 : float
            Frequency of lorentzian (muHz).
        h : float
            Height of the lorentizan (SNR).
        w : float
            Full width of the lorentzian (muHz).

        Returns
        -------
        mode : ndarray
            The SNR as a function frequency for a lorentzian.
        
        """
        
        return h / (1 + 4*(self.nu - nu0)**2 / w**2)

    # -----------------------------
    # Scaling relations
    # -----------------------------
    def compute_A_rms_max(self, L, M, T):
        """
        Compute maximum RMS RV amplitude, A_max [m/s] using scaling relations. (Kjeldsen & Bedding 2011)

        Parameters
        ----------
        L : float
            Stellar luminosity (solar units)
        M : float
            Stellar mass (solar units)
        T : float
            Effective temperature (K)

        Returns
        -------
        A_max : float
            Maximum amplitude
        
        """
        A_max_sun = 0.234        # +/- 0.014 m/s velocity Amplitude of Sun (Kjeldsen & Bedding 1994)
        Tau_sun = 2.88           #days (mode lifetime of Sun Kjeldsen & Bedding 2011)
        T_sun = 5777             #Solar effective temperature (K)
        Tau = Tau_sun * (T/T_sun)**(-3.7)      # Mode lifetime scaling (Chaplin et al. 2009)
        return A_max_sun * ((L * (Tau/Tau_sun)**0.5) / (M**1.5 * (T/T_sun)**2.25))   #Kjeldsen & Bedding 2011

    def compute_env_width(self, T, numax):
        """
        Compute the FWHM of the power envelope (Ball et al. 2018).

        Parameters
        ----------
        T : float
            Effective temperature (K)
        numax : float
            Frequency of maximum power (μHz)

        Returns
        -------
        sig_env : float
            Standard deviation of the Gaussian envelope
        
        """
        
        T_sun = 5777
        FWHM_env = 0.66 * numax**0.88
        if T > T_sun: FWHM_env *= 1 + 6e-4 * (T - T_sun)
        return FWHM_env / (2 * np.sqrt(2 * np.log(2)))

    # -----------------------------
    # Mode linewidths
    # -----------------------------
    def compute_mode_width_1(self, nu, nu_max, T_eff):
        """
        Estimate the linewidth (FWHM) of solar-like oscillation modes at given frequencies (Ball et al. 2018),
        with optional maximum mode width cutoff.
        
        Parameters
        ----------
        nu : ndarray
            Frequencies (μHz) at which to compute the mode linewidths.
        nu_max : float
            Frequency of maximum oscillation power (μHz), a proxy for stellar evolutionary state.
        T_eff : float
            Effective temperature of the star (Kelvin).

        Returns
        -------
        widths : ndarray
            Mode linewidths (μHz) at the input frequencies.
        
        """
        # Simplified calculation using fitted coefficients
        # Returns widths as array
        
        # Coefficients
        a1, b1, c1 = -3.710e0,   1.073e-3,   1.883e-4
        a2, b2, c2 = -7.209e1,   1.543e-2,   9.101e-4
        a3, b3, c3 = -2.266e-1,  5.083e-5,   2.715e-6
        a4, b4, c4 = -2.190e3,   4.302e-1,   8.427e-1
        a5, b5, c5 = -5.639e-1,  1.138e-4,   1.312e-4
        
       

        alpha        = a1 + b1 * T_eff + c1 * nu_max
        gam_alpha    = a2 + b2 * T_eff + c2 * nu_max
        del_gam_dip  = a3 + b3 * T_eff + c3 * nu_max
        nu_dip       = a4 + b4 * T_eff + c4 * nu_max
        W_dip        = a5 + b5 * T_eff + c5 * nu_max
        
        eps = 1e-10
        del_gam_dip = np.maximum(del_gam_dip, eps)
        gam_alpha = np.maximum(gam_alpha, eps)
        nu_ratio = np.maximum(nu / nu_max, eps)
        nu_dip_ratio = np.maximum(nu / nu_dip, eps)
        wdip_ratio = np.maximum(W_dip / nu_max, eps)

        log_ratio = np.log(nu_ratio)
        log_wdip_ratio = np.log(wdip_ratio)
        dip_term = np.log(del_gam_dip) / (1 + (2 * np.log(nu_dip_ratio) / log_wdip_ratio) ** 2)

        width = np.exp(alpha * log_ratio + np.log(gam_alpha) + dip_term)

        return width

    def compute_mode_width_2(self, nu, numax, T_eff, min_width=1.2e-5, a_left=1e-4, p_left=1.2, a_right=3e-4, p_right=1.35, max_width=None):
        
        """
        Compute a asymmetric mode linewidth profile centered on numax,
        used for cool stars (T_eff < 4500 K).

        Parameters
        ----------
        nu : float or ndarray
            Mode frequency (μHz)
        numax : float
            Frequency of maximum oscillation power (μHz)
        T_eff : float
            Effective temperature (K)
        min_width : float
            Minimum mode width at numax (μHz)
        a_left, p_left : float
            Amplitude and power law exponent for the left (ν < ν_max) slope
        a_right, p_right : float
            Amplitude and power law exponent for the right (ν > ν_max) slope
        max_width : float or None
            Maximum allowed mode width (μHz). If None, no max applied.

        Returns
        -------
        width : float or ndarray
            Frequency-dependent mode width (μHz)
        
        """
        
        nu = np.atleast_1d(nu)
        width = np.full_like(nu, T_eff*min_width, dtype=float)  # Base width scaling with T_eff
        width[nu<numax] += a_left * (numax - nu[nu<numax])**p_left   #Scale widths asymmetrically around numax
        width[nu>numax] += a_right * (nu[nu>numax] - numax)**p_right
        if max_width is not None: width = np.minimum(width, max_width) #Apply maximum width cap if specified to avoid unphysically large widths
        return width if width.size > 1 else width[0]

    # -----------------------------
    # Mode power
    # -----------------------------
    def compute_mode_power(self, nu, L, M, T, numax):
        """
        Compute the mode power at given frequency using the RMS amplitude envelope and Gaussian envelope width.

        Parameters
        ----------
        nu : float or ndarray
            Frequency of the mode (μHz)
        L : float
            Stellar luminosity (in L_sun)
        M : float
            Stellar mass (in M_sun)
        T : float
            Effective temperature (K)
        numax : float
            Frequency of maximum oscillation power (μHz)

        Returns
        -------
        power : float or ndarray
            Mode power at frequency nu
        
        """
        
        A_max = self.compute_A_rms_max(L, M, T)
        sigma_env = self.compute_env_width(T, numax)
        return A_max**2 * np.exp(-(nu - numax)**2 / (2*sigma_env**2))

    # -----------------------------
    # Add all modes to spectrum
    # -----------------------------
    def addModes(self, T_eff, L, M, numax, dnu, **kwargs):
        """
        Add oscillation modes (l=0,1,2) with frequency-dependent linewidths and power.
        Returns mode data used to create a .con file.
        
        """
        
        modes = np.zeros_like(self.nu)
        nu0_p = self.asymptotic_nu_p(numax, dnu, **kwargs)
        d02 = dnu * (9.0 / 135.1)
        d01 = dnu/2 - (dnu * (4.5 / 135.1))    #ok for now.

        mode_data = []

        for i, n in enumerate(self.N_p):
            nu_l0 = nu0_p[i]
            nu_l1 = nu_l0 + d01
            nu_l2 = nu_l0 - d02

            #Linewidths: use curved profile for cool stars
            if T_eff < 4500:
                width_l0 = self.compute_mode_width_2(nu_l0, numax, T_eff, max_width=10)
                width_l1 = self.compute_mode_width_2(nu_l1, numax, T_eff, max_width=10)
                width_l2 = self.compute_mode_width_2(nu_l2, numax, T_eff, max_width=10)
            else:
                width_l0 = self.compute_mode_width_1(nu_l0, numax, T_eff)
                width_l1 = self.compute_mode_width_1(nu_l1, numax, T_eff)
                width_l2 = self.compute_mode_width_1(nu_l2, numax, T_eff)

            #Mode Powers
            power_l0 = self.compute_mode_power(nu_l0, L, M, T_eff, numax)
            power_l1 = self.compute_mode_power(nu_l1, L, M, T_eff, numax) * self.vis['V10']
            power_l2 = self.compute_mode_power(nu_l2, L, M, T_eff, numax) * self.vis['V20']

            #Mode Heights
            height_l0 = (np.pi/2) * power_l0 / width_l0
            height_l1 = (np.pi/2) * power_l1 / width_l1
            height_l2 = (np.pi/2) * power_l2 / width_l2

            #Add Lorentzian contributions
            modes += self.lor(nu_l0, height_l0, width_l0)
            modes += self.lor(nu_l1, height_l1, width_l1)
            modes += self.lor(nu_l2, height_l2, width_l2)

            # Store mode info for .con file
            mode_data.extend([
                {"l": 0, "n": n, "nu": nu_l0, "linewidth": width_l0, "power": power_l0},
                {"l": 1, "n": n, "nu": nu_l1, "linewidth": width_l1, "power": power_l1},
                {"l": 2, "n": n, "nu": nu_l2, "linewidth": width_l2, "power": power_l2},
            ])

        return mode_data, modes

#--------------------------------#

def generate_seed_hash(M, T_eff, numax):
    """
    Generates a reproducible integer seed based on M, T_eff, and numax
    using an MD5 hash.
    
    """
    
    s = f"{M:.6f}_{T_eff:.6f}_{numax:.6f}"
    h = hashlib.md5(s.encode()).hexdigest()
    # Uses the first 8 characters of the hash as a base-16 integer
    seed = int(h[:8], 16) % (2**31 - 1)  # ensures it's in 32-bit integer range
    return seed

#--------------------------------#

def create_dir(ID,base_path):
    """
    Creates a directory named after the star ID if it does not already exist.
    
    """
    base_dir = os.path.join(base_path, ID)
    
    try:
        os.makedirs(base_dir, exist_ok=True)
        print(f"Directory '{base_dir}' created (or already exists).")
    except Exception as e:
        print(f"An error occurred while creating directory: {e}")
        
#--------------------------------#

def compute_tau(T_eff, R, M):
    """
    Calculate the granulation timescale (tau) in seconds using the scaling relation from Basu and Chaplin, 2018.
    
    Parameters:
        T (float): Effective temperature in Kelvin.
        R (float): Radius in solar units.
        M (float): Mass in solar units.
    
    Returns:
        tau (float): Granulation timescale in seconds.
    """
    
    T_eff_sun = 5777  # Solar effective temperature in Kelvin
    tau_sun = 200  # Solar granulation timescale in seconds (Reference from Bill)
    
    tau = tau_sun * (R**2 / M) * (T_eff / T_eff_sun)**(1/2)
    #tau = tau_sun * (R**2/M) ** (7/9) * (T_eff_sun/T_eff) ** (23/9)  #method 2 from Basu and Chaplin, 2018
    
    return tau 

#--------------------------------#

def compute_sig_gran(T_eff, R, M):
    """
    Calculate the granulation amplitude (sigma_gran) in m/s using the scaling relation from Basu and Chaplin, 2018.
    
    Parameters:
        T_eff (float): Effective temperature in Kelvin.
        R (float): Radius in solar units.
        M (float): Mass in solar units.
    
    Returns:
        sigma_gran (float): Granulation amplitude in m/s.
    """
    
    T_eff_sun = 5777  # Solar effective temperature in Kelvin
    sigma_gran_sun = 0.8  # Solar granulation amplitude in m/s Meunier et al. 2015 (https://www.aanda.org/articles/aa/abs/2015/11/aa25721-15/aa25721-15.html) 
    R_sun = M_sun = 1  # Solar radius and mass in solar units
    
    #assuming convective velocity (v_c) ∝ sound speed (c_s)
    sigma_gran = sigma_gran_sun * (T_eff/T_eff_sun)**(3/2) * (R/R_sun) * (M/M_sun)**(-1) 
    
    #assuming v_c ∝ M_c * c_s
    #sigma_gran = sigma_gran_sun * (T_eff/T_eff_sun)**(41/9) * (R/R_sun)**(13/9) * (M/M_sun)**(-11/9) 
     
    return sigma_gran

#--------------------------------#

def create_config_file(ID, tau, sig_gran, subfolder, **kwargs):
    """
    Create a customizable configuration file for AADG3 simulations in a specified subfolder.

    Parameters
    ----------
    ID : str
        Identifier for the configuration file (used in filenames).
    tau : float
        Granulation timescale (s) to write in the config file.
    sig_gran : float
        Granulation amplitude to write in the config file.
    subfolder : str
        Path to the folder where the .in file will be saved.
    **kwargs : dict
        Optional overrides for default configuration parameters.
    """

    # ------------------------------------------------------------
    # 1. Ensure the output subfolder exists
    # ------------------------------------------------------------
    # This uses a helper function 'create_dir' to create the folder if it doesn't exist
    #create_dir(ID, base_path=subfolder)

    # ------------------------------------------------------------
    # 2. Define default configuration parameters
    # ------------------------------------------------------------
    # These are typical values for solar-like stars
    config_params = {
        'user_seed': '225618852',         # Random seed for reproducibility
        'cadence': '1d0',                 # Time step (1 second)
        'n_cadences': '6220800',          # Total number of steps (72 days at 1 s)
        'sig': f'{sig_gran}d0',           # Granulation amplitude
        'rho': '0d0',                      # Correlation for red noise (0 = none)
        'tau': f'{tau}d0',                # Granulation timescale in seconds
        'n_relax': '86400d0',             # Relaxation steps before data collection (1 day)
        'n_fine': '50',                    # Resolution factor for interpolation
        'inclination': '90d0',            # Edge-on inclination
        'p1': '1.35d0', 'p2': '1.02d0', 'p3': '0.48d0',  # Mode visibility factors (l=1,2,3)
        'add_granulation': '.false.',     # Include granulation signal
        'modes_filename': f'{ID}.con',   # Input modes file
        'rotation_filename': '',          # Optional rotation file
        'output_filename': f'{ID}.asc'   # Output file
    }

    # ------------------------------------------------------------
    # 3. Override defaults with any user-supplied parameters
    # ------------------------------------------------------------
    # User can pass in any valid config parameter as a keyword argument
    for key, value in kwargs.items():
        if key in config_params:
            config_params[key] = str(value)
        else:
            raise KeyError(f"Unknown configuration parameter: '{key}'")

    # ------------------------------------------------------------
    # 4. Define the file path for the .in file
    # ------------------------------------------------------------
    infile_path = os.path.join(subfolder, f"{ID}.in")

    # ------------------------------------------------------------
    # 5. Generate the content of the .in file
    # ------------------------------------------------------------
    # Uses formatted string to insert all parameters
    infile_content = f"""\
    ! This input produces data for model {ID}. Frequencies, linewidths, and amplitudes were modeled based on stellar parameters and a constant rotation rate of 0 µHz.
    
    &controls
    user_seed = {config_params['user_seed']}
    cadence = {config_params['cadence']}
    n_cadences = {config_params['n_cadences']}
    sig = {config_params['sig']}
    rho = {config_params['rho']}
    tau = {config_params['tau']}
    n_relax = {config_params['n_relax']}
    n_fine = {config_params['n_fine']}
    inclination = {config_params['inclination']}
    p(1) = {config_params['p1']}
    p(2) = {config_params['p2']}
    p(3) = {config_params['p3']}
    add_granulation = {config_params['add_granulation']}
    modes_filename = '{config_params['modes_filename']}'
    rotation_filename = '{config_params['rotation_filename']}'
    output_filename = '{config_params['output_filename']}'
    /
    """

    # ------------------------------------------------------------
    # 6. Write content to file
    # ------------------------------------------------------------
    with open(infile_path, 'w') as file:
        file.write(infile_content)

    # Optional: print confirmation
    # print(f"Configuration file '{infile_path}' created.")


#--------------------------------#

def process_single_realisation(i, ID, N, M_samples, T_samples, R_samples, L_samples, numax_samples, dnu_samples, NU_GRID, base_path, **config_overrides):
    """
    Generate a single Monte Carlo realisation: oscillation modes, PSD, and AADG3 config file.

    Parameters
    ----------
    i : int
        Index of the realisation.
    ID : str
        Star identifier.
    N : int
        Total number of realisations.
    M_samples, T_samples, R_samples, L_samples : np.ndarray
        Stellar parameters arrays.
    numax_samples, dnu_samples : np.ndarray
        Seismic parameters arrays.
    NU_GRID : np.ndarray
        Frequency grid.
    base_path : str
        Base directory where star folders and realisations are saved.
    **config_overrides : dict
        Optional overrides for the config file.
    """

    # Unique ID and absolute subfolder path for this realisation
    this_ID = f"{ID}_MC_{i+1}"
    star_folder = os.path.join(base_path, str(ID).replace("/", "_").replace(" ", "_"))
    subfolder = os.path.join(star_folder, f"MC_{i+1}")
    os.makedirs(subfolder, exist_ok=True)

    # Generate oscillation modes
    m = model(NU_GRID)
    model_data, modes = m.addModes(
        T_eff=T_samples[i],
        L=L_samples[i],
        M=M_samples[i],
        numax=numax_samples[i],
        dnu=dnu_samples[i],
    )

    # Save modes to a .con file
    mode_array = np.array([[d["l"], d["n"], d["nu"], d["linewidth"], d["power"], 0] for d in model_data])
    np.savetxt(os.path.join(subfolder, f"{this_ID}.con"), mode_array, fmt="%d %d %.1f %.1f %.1f %.1f")

    # Save synthetic power spectrum density (PSD) to CSV
    pd.DataFrame({"nu": NU_GRID, "power": modes}).to_csv(
        os.path.join(subfolder, f"{this_ID}_PSD.csv"), index=False
    )

    # Generate reproducible random seed from stellar parameters
    user_seed = generate_seed_hash(M_samples[i], T_samples[i], numax_samples[i])

    # Create the configuration file for AADG3
    create_config_file(
        ID=this_ID,
        tau=compute_tau(T_samples[i], R_samples[i], M_samples[i]),
        sig_gran=compute_sig_gran(T_samples[i], R_samples[i], M_samples[i]),
        subfolder=subfolder,
        user_seed=str(user_seed),
        **config_overrides
    )

#--------------------------------#

def run_all_realisations_parallel(ID, base_path, NU_GRID, max_workers=4, **config_overrides):
    """
    Run all Monte Carlo realisations for a given star in parallel,
    using the parameters saved in the star's CSV file.

    Parameters
    ----------
    ID : str
        Star identifier. Also used to locate the star's folder and CSV file.
    base_path : str
        Base directory where the star's folder and output CSV are located.
    NU_GRID : np.ndarray
        Frequency grid for the PSD used in `process_single_realisation`.
    max_workers : int, optional
        Number of parallel workers to use. Default is 4.
    **config_overrides : dict
        Optional keyword arguments to override default config parameters.
    """
    
    # Ensure the folder name is safe
    main_folder_ID = str(ID).replace("/", "_").replace(" ", "_")
    main_folder= os.path.join(base_path, main_folder_ID)
    csv_path = os.path.join(main_folder, f"{main_folder_ID}_MC_parameters.csv")
    
    # Check that the CSV exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"MC parameters CSV not found for {ID}: {csv_path}")
    
    # Load Monte Carlo parameters
    params_df = pd.read_csv(csv_path)
    N = len(params_df)  # number of realisations

    # Extract arrays for each parameter
    M_samples = params_df['M'].values
    R_samples = params_df['R'].values
    T_samples = params_df['Teff'].values
    L_samples = params_df['L'].values
    numax_samples = params_df['numax'].values
    dnu_samples = params_df['dnu'].values

    # Run all realisations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_realisation,
                i, ID, N,
                M_samples=M_samples,
                T_samples=T_samples,
                R_samples=R_samples,
                L_samples=L_samples,
                numax_samples=numax_samples,
                dnu_samples=dnu_samples,
                NU_GRID=NU_GRID,
                base_path=base_path,
                **config_overrides
            )
            for i in range(N)
        ]
        # Wait for all to complete
        for future in futures:
            future.result()
            
#--------------------------------#

def run_aadg3_single(ID, base_path, aadg3_path, j):
    """
    Run a single AADG3 simulation for the j-th Monte Carlo realisation of a star.

    Parameters
    ----------
    ID : str
        Star identifier (used for folder and file naming).
    base_path : str
        Base directory where star folders are stored.
    aadg3_path : str
        Path to the AADG3 executable.
    j : int
        Index of the Monte Carlo realisation (0-based).
    """

    # ------------------------------------------------------------
    # 1. Build the folder paths and input file name
    # ------------------------------------------------------------
    safe_ID = str(ID).replace(" ", "_").replace("/", "_")
    star_dir = os.path.join(base_path, safe_ID)                 # e.g., base_path/StarName
    star_dir_2 = os.path.join(star_dir, f"MC_{j+1}")      # Subfolder for this MC realisation
    input_file = f"{ID}_MC_{j+1}.in"                      # Corresponding .in configuration file

    # ------------------------------------------------------------
    # 2. Run AADG3 using subprocess
    # ------------------------------------------------------------
    try:
        result = subprocess.run(
            [aadg3_path, input_file],  # Command: executable + input file
            cwd=star_dir_2,            # Run in the folder for this realisation
            check=True,                # Raise exception if process fails
            stdout=subprocess.PIPE,    # Capture standard output
            stderr=subprocess.PIPE     # Capture standard error
        )

        # If successful, print confirmation
        print(f"[✓] AADG3 run for {ID} MC_{j+1} completed.")

    # ------------------------------------------------------------
    # 3. Handle errors
    # ------------------------------------------------------------
    except subprocess.CalledProcessError as e:
        # AADG3 failed (non-zero exit status)
        print(f"[✗] AADG3 run for {ID} MC_{j+1} failed: {e.stderr.decode()}")

    except Exception as ex:
        # Catch all other exceptions (e.g., file not found)
        print(f"[!] AADG3 run for {ID} MC_{j+1} error: {ex}")
        
#--------------------------------#

def run_all_aadg3_parallel(ID, base_path, aadg3_path, N, max_workers=4):
    """
    Run AADG3 for all N Monte Carlo realisations of a star in parallel.

    Parameters
    ----------
    ID : str
        Star identifier (folder name).
    base_path : str
        Base directory where the star's folder is located.
    aadg3_path : str
        Path to the AADG3 executable.
    N : int
        Number of Monte Carlo realisations to run.
    max_workers : int, optional
        Maximum number of parallel threads to use (default 4).
    """

    # Ensure we do not exceed available CPU cores
    # max_workers = min(max_workers, multiprocessing.cpu_count())

    # Run all realisations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_aadg3_single, ID, base_path, aadg3_path, j)
            for j in range(N)
        ]

        # Wait for all to finish
        for future in futures:
            future.result()

#--------------------------------#

def process_realisation_star(filepath, ints, T_fixed):
    """
    Compute the standard deviation of binned means for a range of 
    integration times (ints). Each integration time defines the bin length
    used to segment the AADG3 time series and compute mean flux values.

    Parameters
    ----------
    filepath : str
        Path to the .asc file containing the AADG3 realisation (single-column time series).

    ints : array_like
        Array of desired integration times (in seconds).

    T_fixed : float
        Total time window (seconds) over which each integration should be computed.
        The window is subdivided into bins of equal length for each integration time.

    Returns
    -------
    std_dev : ndarray or None
        Array of standard deviations—one per integration time.
        Returns None if input data contain NaNs.
    """

    # ---------------------------------------------------------
    # Load the AADG3 time series (1D array)
    # ---------------------------------------------------------
    ts = np.loadtxt(filepath)

    # If the time series contains NaNs, skip this realisation
    if np.any(np.isnan(ts)):
        return None

    total_len = len(ts)
    std_dev = []   # Output list: one std-dev per integration time

    # ---------------------------------------------------------
    # Loop through each desired integration time
    # ---------------------------------------------------------
    for exposure in ints:

        # Number of bins to place inside the fixed total window
        npts = round(T_fixed / exposure)

        # If the requested integration time is too large to form a bin
        if npts == 0:
            std_dev.append(np.nan)
            continue

        # Exact bin width needed so that npts bins fill T_fixed exactly
        # (Not necessarily equal to 'exposure' due to rounding)
        exact_exposure = T_fixed / npts

        # Convert bin width (seconds) into number of samples
        seg_len = int(exact_exposure)

        # Total number of samples required to form npts bins
        needed_len = npts * seg_len

        # If the time series is too short for this integration time, skip
        if needed_len > total_len:
            std_dev.append(np.nan)
            continue

        # ---------------------------------------------------------
        # Reshape the first needed_len samples into an array of bins
        # dimensions: (npts bins × seg_len samples per bin)
        # ---------------------------------------------------------
        reshaped = ts[:needed_len].reshape(npts, seg_len)

        # Compute the mean value for each bin
        seg_means = reshaped.mean(axis=1)

        # Compute standard deviation of the bin means
        std_dev.append(seg_means.std())

    # Convert list to array
    return np.array(std_dev)

#--------------------------------#

def find_numax(ints, std_dev, order=4):
    """
    Estimate nu_max by identifying the first local minimum in the
    std-dev vs integration-time curve, and refining its location
    using a parabola fit in log-log space.

    Parameters
    ----------
    ints : array-like
        Integration times (seconds) used to compute std_dev.

    std_dev : array-like
        Standard deviation values for each integration time.

    order : int
        Number of neighboring points on each side required for a
        point to be considered a local minimum (scipy.signal.argrelmin).

    Returns
    -------
    tuple:
        (nu_max_uHz, nu_max_minutes, coeffs, (x_win, y_win))

        nu_max_uHz      : Estimated nu_max in microhertz.
        nu_max_minutes  : Integration time (in minutes) that corresponds  
                          to the variability minimum.
        coeffs          : Parabolic fit coefficients in log-log space (a, b, c),
                          or None if no fit was possible.
        (x_win, y_win)  : The log-log window used for fitting,
                          or None if no fit was performed.
    """

    # ---------------------------------------------------------
    # 1. Remove NaNs from the data
    # ---------------------------------------------------------
    valid = ~np.isnan(std_dev)
    if not np.any(valid):
        # All values are invalid → cannot estimate ν_max
        return None

    clean_ints = ints[valid]
    clean_std  = std_dev[valid]

    # ---------------------------------------------------------
    # 2. Identify local minima in the std-dev curve
    # ---------------------------------------------------------
    minima_indices = argrelmin(clean_std, order=order)[0]

    # If no local minima are found, fall back to the global minimum
    if len(minima_indices) == 0:
        first_min_index = np.argmin(clean_std)
    else:
        # Use the *first* local minimum as the ν_max indicator
        first_min_index = minima_indices[0]

    # ---------------------------------------------------------
    # 3. Check if there are enough points around the minimum
    #    for a parabola fit (need 2 points on each side)
    # ---------------------------------------------------------
    padding = 2
    if not (padding <= first_min_index <= len(clean_ints) - padding - 1):

        # Not enough neighbors → return ν_max using raw minimum value
        tp_mins = clean_ints[first_min_index] / 60           # Convert seconds → minutes
        tp_uHz  = 1 / (tp_mins * 60) * 1e6                   # Convert min → μHz via ν = 1/T

        return tp_uHz, tp_mins, None, None

    # ---------------------------------------------------------
    # 4. Extract window around minimum and fit a parabola in log-log space
    # ---------------------------------------------------------

    # log10 is unnecessary—natural log is fine, since polyfit only sees relative scaling
    x_win = np.log(clean_ints[first_min_index - padding : first_min_index + padding + 1] / 60)
    y_win = np.log(clean_std[first_min_index - padding : first_min_index + padding + 1])

    # Quadratic fit → y = ax² + bx + c
    coeffs = np.polyfit(x_win, y_win, 2)

    # ---------------------------------------------------------
    # 5. Find the vertex (minimum) of the parabola
    # ---------------------------------------------------------
    # For y = ax² + bx + c, minimum is at x = -b / (2a)
    min_x = -coeffs[1] / (2 * coeffs[0])

    # Convert log(x) → x, and seconds → minutes
    min_x_mins = np.exp(min_x)

    # Convert the characteristic time scale T (in minutes) to ν_max in μHz:
    # ν = 1 / (T * 60)      → convert minutes to seconds
    # multiply by 1e6 to convert Hz to μHz
    min_x_uHz = 1 / (min_x_mins * 60) * 1e6

    # ---------------------------------------------------------
    # 6. Return results
    # ---------------------------------------------------------
    return min_x_uHz, min_x_mins, coeffs, (x_win, y_win)

#--------------------------------#

def analyze_all_realisations(base_path, ID, ints, T_fixed, max_realisations=None):
    """
    Analyze AADG3 realisations without plotting, and save results to .npz.

    Parameters
    ----------
    base_path : str
        Path to the directory containing the star folder <ID>.

    ID : str
        Name of the star folder inside base_path.

    ints : array_like
        Integration times (seconds) used for rolling std-dev.

    T_fixed : float
        Total integration window (seconds) used for std-dev.

    max_realisations : int, optional
        Limit the number of MC_* folders processed.

    Returns
    -------
    None
        Saves results to base_path/ID/ID.npz.
    """
    ID_f = str(ID).replace(" ", "_")
    # Full path to the star folder
    star_folder = os.path.join(base_path, ID_f)

    # ------------------------------------------------------------------
    # Locate and sort MC_* subfolders
    # ------------------------------------------------------------------
    realisation_paths = sorted(
        glob.glob(os.path.join(star_folder, "MC_*")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )

    if max_realisations is not None:
        realisation_paths = realisation_paths[:max_realisations]

    all_std_devs = []
    numaxes = []
    used_count = 0
    skipped_count = 0

    # ------------------------------------------------------------------
    # Loop over MC_* folders
    # ------------------------------------------------------------------
    for real_path in realisation_paths:

        # Folder is MC_1, MC_2, ...
        mc_name = os.path.basename(real_path)

        # Expected ASC file name:
        # Example: <ID>_MC_1.asc
        asc_filename = f"{ID}_{mc_name}.asc"

        filepath = os.path.join(real_path, asc_filename)

        # Compute std-dev curve
        std_dev = process_realisation_star(filepath, ints, T_fixed)

        if std_dev is None or np.all(np.isnan(std_dev)):
            skipped_count += 1
            continue

        used_count += 1
        all_std_devs.append(std_dev)

        # Estimate ν_max
        nu_max, _, _, _ = find_numax(ints, std_dev)
        if nu_max is not None:
            numaxes.append(nu_max)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    all_std_devs = np.array(all_std_devs)
    mean_std_dev = np.nanmean(all_std_devs, axis=0)
    std_std_dev = np.nanstd(all_std_devs, axis=0)

    numaxes = np.array(numaxes)
    mean_numax = np.mean(numaxes)
    std_numax = np.std(numaxes)
    mean_numax_min = 1 / (mean_numax * 1e-6) / 60  # μHz → min

    # Save to base_path/ID/ID.npz
    save_path = os.path.join(star_folder, f"{ID}.npz")

    np.savez_compressed(
        save_path,
        ints=ints,
        all_std_devs=all_std_devs,
        mean_std_dev=mean_std_dev,
        std_std_dev=std_std_dev,
        numaxes=numaxes,
        mean_numax=mean_numax,
        std_numax=std_numax,
        mean_numax_min=mean_numax_min,
    )

    print(f"Used {used_count} realisations. Skipped {skipped_count}.")
    print(f"Saved results to {save_path}")
    
#-----------------------------------------#
#Data plotting and visualisation functions
#-----------------------------------------#

def load_star_data(sample, star_index, mc_index, base_path):
    """
    Load star data from Monte Carlo outputs, including PSD and mode configuration.

    This function reads the following:
    - Power spectrum CSV file (PSD)
    - Mode configuration (.con) file
    - Monte Carlo parameters CSV
    for a given star and a specific MC realisation.

    Parameters
    ----------
    sample : pandas.DataFrame
        DataFrame containing the list of stars and their metadata.
    star_index : int
        Index of the star in the sample DataFrame to load.
    mc_index : int
        Monte Carlo realisation number (1-based).
    base_path : str
        Base directory where star folders and MC outputs are located.

    Returns
    -------
    dict
        Dictionary containing:
        - ID: Base star name
        - ID_MC: Unique ID for this MC realisation
        - nu1: Frequency array from PSD CSV (μHz)
        - power1: Power spectrum density from PSD CSV [(m/s)^2/μHz]
        - l: Angular degree array from .con file
        - n: Radial order array from .con file
        - nu2: Mode frequencies from .con file (μHz)
        - linewidth: Mode linewidths from .con file (μHz)
        - power2: RMS power of modes from .con file [(m/s)^2]
        - numax: Frequency of maximum oscillation power (μHz) from MC
        - dnu: Large frequency separation (μHz) from MC
        - Teff: Effective temperature (K) from MC
    """

    # ----------------------------------------------------------------------
    # 1. Get star name and generate unique ID for this MC realisation
    # ----------------------------------------------------------------------
    row = sample.loc[star_index]           # Select the star row from the DataFrame
    ID = row['primary_name']               # Name of the star
    ID_f = ID.replace(' ', '_')              # Safe folder name   
    ID_MC = f"{ID}_MC_{mc_index}"     # Unique ID for this MC realisation

    # ----------------------------------------------------------------------
    # 2. Construct file paths
    # ----------------------------------------------------------------------
    MC_folder = os.path.join(base_path,ID_f, f"MC_{mc_index}")     # Folder containing MC folders
    file_path1 = os.path.join(MC_folder, f"{ID_MC}_PSD.csv")          # PSD CSV path
    file_path2 = os.path.join(MC_folder, f"{ID_MC}.con")              # Mode configuration file path
    file_path3 = os.path.join(base_path,ID_f,f"{ID_f}_MC_parameters.csv")  # MC parameters CSV

    # ----------------------------------------------------------------------
    # 3. Verify that required files exist
    # ----------------------------------------------------------------------
    for fpath in [file_path1, file_path2, file_path3]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")
        
        
    # ----------------------------------------------------------------------
    # 4. Load data from files
    # ----------------------------------------------------------------------
    csv_file = pd.read_csv(file_path1)          # Load PSD CSV
    con_data = np.loadtxt(file_path2)           # Load .con file
    mc_params = pd.read_csv(file_path3)     # Load MC parameters CSV

    # Sanity check for empty files
    if csv_file.empty or con_data.size == 0:
        raise ValueError("Loaded files are empty.")

    # ----------------------------------------------------------------------
    # 5. Extract Monte Carlo parameters for this realisation
    # ----------------------------------------------------------------------
    mc_row = mc_params[mc_params['realisation'] == mc_index]
    if mc_row.empty:
        raise ValueError(f"MC realisation {mc_index} not found in parameters file")
    mc_row = mc_row.iloc[0]                     # Take the first (and should be only) matching row

    # ----------------------------------------------------------------------
    # 6. Extract relevant arrays and parameters
    # ----------------------------------------------------------------------
    data = {
        'ID': ID,
        'ID_base': ID_MC,
        'nu1': csv_file['nu'].values,          # Frequency array from PSD CSV
        'power1': csv_file['power'].values,    # Power spectrum density from PSD CSV
        'l': con_data[:, 0],                   # Angular degree of modes
        'n': con_data[:, 1],                   # Radial order of modes
        'nu2': con_data[:, 2],                 # Mode frequencies
        'linewidth': con_data[:, 3],           # Linewidths of modes
        'power2': con_data[:, 4],              # RMS power of modes
        'numax': mc_row['numax'],              # ν_max from MC
        'dnu': mc_row['dnu'],                  # Δν from MC
        'Teff': mc_row['Teff'],                # Effective temperature from MC
    }

    return data

#--------------------------------#

def plot_oscillation_modes(data, xlim=(0, 10000), figsize=(10, 6), fontsize=16):
    """
    Plot the power spectrum (PSD) of stellar oscillation modes.

    Parameters
    ----------
    data : dict
        Star data from load_star_data(), expects 'nu1', 'power1', 'ID'.
    xlim : tuple
        X-axis (frequency) limits in μHz.
    figsize : tuple
        Figure size.
    fontsize : int
        Font size for labels and title.
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Plot PSD
    plt.plot(data['nu1'], data['power1'], 
             label=f'Oscillation Modes for {data["ID_base"]}', color='purple')

    # Labels and title
    plt.xlabel("Frequency (μHz)", fontsize=fontsize)
    plt.ylabel("Power (m/s)$^2$/μHz", fontsize=fontsize)
    plt.title(f'Oscillation Modes for {data["ID_base"]}', fontsize=fontsize+2)

    # Axis limits, legend, layout
    plt.xlim(xlim)
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()

#--------------------------------#

def plot_linewidth_vs_frequency(data, figsize=(10, 6), fontsize=16):
    """
    Plot mode linewidths as a function of scaled frequency.

    Parameters
    ----------
    data : dict
        Star data from load_star_data(), expects 'nu2', 'linewidth', 'numax', 'dnu', 'ID'.
    figsize : tuple
        Figure size (width, height).
    fontsize : int
        Font size for labels and title.
    """
    # Scale frequency relative to ν_max and Δν
    scaled_freq = (data['nu2'] - data['numax']) / data['dnu']

    # Create figure and plot
    plt.figure(figsize=figsize)
    plt.plot(scaled_freq, data['linewidth'], color='purple')
    plt.xlabel('$(\\nu - \\nu_{\\rm max})/\\Delta\\nu$', fontsize=fontsize)
    plt.ylabel('Mode Width [µHz]', fontsize=fontsize)
    plt.title(f'Mode Width vs Radial Order for {data["ID_base"]}', fontsize=fontsize+2)
    plt.tight_layout()
    plt.show()
    
#--------------------------------#

def epanechnikov_weights(size=5):
    """
    Generate normalized Epanechnikov kernel weights for smoothing.

    Parameters
    ----------
    size : int
        Window size (must be odd).

    Returns
    -------
    np.ndarray
        Normalized 1D array of kernel weights.
    """
    assert size % 2 == 1, "Window size must be odd"
    radius = size // 2
    x = np.arange(-radius, radius + 1)  # positions relative to center
    u = x / radius                       # normalize to [-1, 1]
    weights = 0.75 * (1 - u**2)         # Epanechnikov kernel
    weights /= weights.sum()             # normalize to sum=1
    return weights

#--------------------------------#

def smooth_epanechnikov(y, window_size=9):
    """
    Apply Epanechnikov kernel smoothing to a 1D array.

    Parameters
    ----------
    y : np.ndarray
        Input data array to smooth.
    window_size : int
        Size of smoothing window (must be odd).

    Returns
    -------
    np.ndarray
        Smoothed data array (same length as input).
    """
    weights = epanechnikov_weights(window_size)        # Get kernel weights
    return np.convolve(y, weights, mode='same')       # Apply smoothing

#--------------------------------#

def plot_amplitudes_by_degree(data, figsize=(10, 6), fontsize=16, smooth=True, window_size=9):
    """
    Plot RMS amplitudes of oscillation modes by angular degree l.

    Parameters
    ----------
    data : dict
        Star data from load_star_data(), expects 'nu2', 'power2', 'l', 'numax', 'dnu', 'ID'.
    figsize : tuple
        Figure size (width, height).
    fontsize : int
        Font size for labels and title.
    smooth : bool
        Apply smoothing to amplitude curves.
    window_size : int
        Window size for smoothing.
    """
    # Scale frequencies relative to ν_max and Δν
    scaled_freq = (data['nu2'] - data['numax']) / data['dnu']
    amplitudes = np.sqrt(data['power2'])  # Convert PSD to RMS amplitude

    plt.figure(figsize=figsize)
    
    # Plot amplitudes for each l value with optional smoothing
    colors = {0:'blue', 1:'green', 2:'red', 3:'orange', 4:'purple'}
    for l_val in np.unique(data['l']):
        mask = data['l'] == l_val
        if np.any(mask):
            nu_vals = data['nu2'][mask]
            amp_vals = amplitudes[mask]
            sort_idx = np.argsort(nu_vals)
            x = (nu_vals[sort_idx] - data['numax']) / data['dnu']
            y = amp_vals[sort_idx]
            if smooth and len(y) >= window_size:
                y = smooth_epanechnikov(y, window_size=window_size)
            plt.plot(x, y, label=f'l = {int(l_val)}', color=colors.get(int(l_val), 'black'))
    
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('$(\\nu - \\nu_{\\rm max}) / \\Delta\\nu$', fontsize=fontsize)
    plt.ylabel('RMS amplitude [(m/s)]', fontsize=fontsize)
    plt.title(f'Radial Mode Amplitudes for {data["ID_base"]}', fontsize=fontsize+2)
    plt.legend()
    plt.tight_layout()
    plt.show()

#--------------------------------#

def plot_echelle_diagram(data, figsize=(10, 6), fontsize=16):
    """
    Plot an Échelle diagram and compute Δν₀₂ (small separation between l=0 and l=2 modes).

    Parameters
    ----------
    data : dict
        Star data from load_star_data(), expects 'nu2', 'l', 'numax', 'dnu', 'ID'.
    figsize : tuple
        Figure size (width, height).
    fontsize : int
        Font size for labels and title.

    Returns
    -------
    dict
        Δν₀₂ analysis results:
        - 'mean_delta_nu_02': mean small separation (μHz)
        - 'delta_nu_02_values': list of individual Δν₀₂ values
        - 'pairs': l=0/l=2 mode frequency pairs used
    """
    # Modulate frequencies by Δν and assign colors by angular degree
    mod_freq = np.array(data['nu2']) % data['dnu']
    colors = [ {0:'red',1:'green',2:'blue'}.get(lv,'gray') for lv in data['l'] ]

    # Plot Échelle diagram
    plt.figure(figsize=figsize)
    plt.scatter(mod_freq, data['nu2'], c=colors, edgecolor='k', s=80)
    plt.xlabel(f'Frequency mod Δν = {data["dnu"]:.2f} μHz', fontsize=fontsize)
    plt.ylabel('Frequency [μHz]', fontsize=fontsize)
    plt.title(f'Échelle Diagram for {data["ID_base"]}', fontsize=fontsize+2)
    plt.grid(True)

    # Legend for l values
    plt.legend(handles=[Line2D([0],[0], marker='o', color='w', label=f'l={i}',
                               markerfacecolor=c, markersize=10, markeredgecolor='k')
                        for i,c in zip([0,1,2],['red','green','blue'])],
               title="Angular Degree (l)", loc='center left', bbox_to_anchor=(1,0.4))

    # νmax line
    plt.axhline(data['numax'], color='purple', linestyle='--', linewidth=2)
    plt.text(data['dnu']/2, data['numax']+5, 'νmax', color='purple', ha='center')

    # Compute Δν₀₂: pair l=0 with nearest lower l=2 modes
    nu_l0 = np.sort(data['nu2'][data['l']==0])
    nu_l2 = np.sort(data['nu2'][data['l']==2])
    delta_nu_02, pairs = [], []
    for nu0 in nu_l0:
        lower_l2 = nu_l2[nu_l2<nu0]
        if len(lower_l2)>0:
            closest_l2 = lower_l2[-1]
            delta_nu_02.append(nu0 - closest_l2)
            pairs.append((closest_l2, nu0))

    # Store results
    results = {'mean_delta_nu_02': np.mean(delta_nu_02) if delta_nu_02 else None,
               'delta_nu_02_values': delta_nu_02,
               'pairs': pairs}

    # Annotate one representative Δν₀₂
    if pairs:
        mod_l2_0, mod_l0_0 = pairs[0][0]%data['dnu'], pairs[0][1]%data['dnu']
        plt.plot([mod_l2_0, mod_l0_0], [pairs[0][0], pairs[0][1]], 'k--', alpha=0.6)
        plt.text(0.5*(mod_l2_0+mod_l0_0), 0.5*(pairs[0][0]+pairs[0][1]),
                 f'Δν₀₂ ≈ {results["mean_delta_nu_02"]:.2f} μHz',
                 fontsize=10, color='black', rotation=90)

    plt.tight_layout(rect=[0,0,0.8,1])
    plt.show()

    return results

#--------------------------------#

def read_cadence_from_infile(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            if 'cadence' in line:
                return float(line.split('=')[1].split()[0].replace('d0', ''))

#--------------------------------#

def plot_synthetic_time_series(sample, index, realisation, base_path, fontsize=16):
    """
    Plot a synthetic radial velocity (RV) time series for a given star MC realisation.

    Parameters
    ----------
    sample : pandas.DataFrame
        Star metadata, must include 'primary_name' and 'numax'.
    index : int
        Row index of the star in the DataFrame.
    realisation : int
        Monte Carlo realisation number (1-based).
    base_path : str
        Directory containing AADG3 outputs.
    fontsize : int
        Font size for labels and title.

    Returns
    -------
    str
        Primary name of the star plotted.
    """
    
    # Get star name
    row = sample.loc[index]
    ID = row['primary_name']                # Display name
    ID_f = ID.replace(' ', '_')             # Folder-safe name

    # File paths
    asc_path = os.path.join(base_path, f"{ID_f}/MC_{realisation}/{ID}_MC_{realisation}.asc")
    infile_path = os.path.join(base_path, f"{ID_f}/MC_{realisation}/{ID}_MC_{realisation}.in")

    # Load flux and cadence
    flux = np.loadtxt(asc_path)
    cadence = read_cadence_from_infile(infile_path)

    # Construct time axis
    n_points = len(flux)
    time = np.arange(n_points) * cadence
    
    # Dynamic y-axis limits (110% of variation)
    y_min, y_max = np.min(flux), np.max(flux)
    y_center = 0.5 * (y_min + y_max)
    y_range = (y_max - y_min) * 0.55
    y_lower, y_upper = y_center - y_range, y_center + y_range

    # Plot time series
    plt.figure(figsize=(10, 4))
    plt.plot(time, flux, lw=0.5, color='purple')
    plt.xlabel("Time (days)", fontsize=fontsize)
    plt.ylabel("RV [m/s]", fontsize=fontsize)
    plt.title(f"Synthetic time series for {ID}_MC_{realisation}", fontsize=fontsize+2)
    plt.ylim(y_lower, y_upper)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#--------------------------------#

def plot_std_vs_ints(npz_path, fontsize=16):
    """
    Plot RMS amplitude vs integration (exposure) time from a saved .npz analysis.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing Monte Carlo RMS results.
    fontsize : int
        Font size for labels and title.

    Notes
    -----
    - Plots all individual MC realisations in light green.
    - Overlays the mean ±1σ spread in purple.
    - Highlights the fitted ν_max and its uncertainty in red.
    - Uses logarithmic axes for exposure time and RMS amplitude.
    """
    
    # Load data from .npz file
    data = np.load(npz_path)
    ints = data['ints']                  # Integration times
    all_std_devs = data['all_std_devs']  # RMS of all individual realisations
    mean_std_dev = data['mean_std_dev']  # Mean RMS
    std_std_dev = data['std_std_dev']    # Std of RMS
    mean_numax = data['mean_numax']      # Fitted ν_max (μHz)
    std_numax = data['std_numax']        # Std of ν_max
    mean_numax_min = data['mean_numax_min']  # ν_max in minutes
    sigma_mins = std_numax * mean_numax_min**2 / (1e6 / 60)  # ±1σ in minutes

    star_name = os.path.splitext(os.path.basename(npz_path))[0]

    plt.figure(figsize=(10, 6))

    # Plot all individual MC realisations
    for std_dev in all_std_devs:
        plt.plot(ints / 60, std_dev, color='green', alpha=0.3, lw=0.7)
    plt.plot([], [], color='green', alpha=0.3, label=f'({len(all_std_devs)}) individual realisations')

    # Plot mean ±1σ envelope
    plt.fill_between(
        ints / 60,
        mean_std_dev - std_std_dev,
        mean_std_dev + std_std_dev,
        color='purple',
        alpha=0.8,
        label='±1σ spread'
    )
    plt.plot(ints / 60, mean_std_dev, color='black', lw=1, label='Mean realisation')

    # Highlight fitted ν_max
    plt.axvline(mean_numax_min, color="red", linestyle="--", label=f"Fitted ν_max: {mean_numax:.2f} μHz")
    plt.fill_betweenx(
        y=plt.ylim(),
        x1=mean_numax_min - sigma_mins,
        x2=mean_numax_min + sigma_mins,
        color='red',
        alpha=0.2,
        label=f'±1σ ν_max = ± {std_numax:.2f} μHz'
    )

    # Logarithmic axes
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposure time (minutes)', fontsize=fontsize)
    plt.ylabel('RMS Amplitude (m/s)', fontsize=fontsize)
    plt.title(f'RMS Amplitude vs Exposure Time for {star_name}', fontsize=fontsize+2)
    plt.legend()
    plt.tight_layout()
    plt.show()

#--------------------------------#

def plot_sim(
    sample,
    star_name,
    mc_index=0,
    base_path=None,
    generate_oscillation=True,
    generate_linewidth=True,
    generate_amplitude=True,
    generate_echelle=True,
    generate_time_series=True,
    generate_from_npz=True,
    xlim=(0, 10000),
    figsize=(10, 6)
):
    """
    Wrapper function to read data and generate plots for a given star.

    Parameters
    ----------
    sample : pandas.DataFrame
        Full sample table loaded from the input CSV. Must include the column
        'primary_name' for star identification.

    star_name : str
        Name of the target star exactly as listed in `sample['primary_name']`.

    mc_index : int, optional
        Monte Carlo realisation index to load for visualisation of singular time series.

    base_path : str, optional
        Base directory containing star-specific folders and AADG3 outputs.
        Required for time-series plotting and .npz access.

    generate_oscillation : bool
        If True, produce plots of oscillation mode frequencies and amplitudes.

    generate_linewidth : bool
        If True, plot linewidths as a function of frequency.

    generate_amplitude : bool
        If True, plot mode amplitudes grouped by angular degree.

    generate_echelle : bool
        If True, compute and plot the echelle diagram based on the radial and
        non-radial mode frequencies. Returns Δν₀₂ results.

    generate_time_series : bool
        If True, construct and plot synthetic RV or photometric time series
        based on the realisation's AADG3 configuration.

    generate_from_npz : bool
        If True, load existing simulation results from a .npz file and plot
        the stored power spectrum or time series.

    xlim : tuple
        Frequency axis limits for oscillation-mode plots (e.g., 0–10,000 μHz).

    figsize : tuple
        Figure size for matplotlib plots.

    Returns
    -------
    dict
        {
            "data": data_dict,
            "echelle_results": echelle_results or None
        }
        where `data_dict` contains all loaded mode and scaling results for the
        selected MC realisation, and `echelle_results` stores Δν₀₂ estimates if
        computed.
    """

    # ----------------------------------------------------------------------
    # 1. Identify the target star in the sample table
    # ----------------------------------------------------------------------
    star_index = sample[sample['primary_name'] == star_name].index[0]

    # ----------------------------------------------------------------------
    # 2. Load all data relevant to this star and MC realisation.
    # ----------------------------------------------------------------------
    data = load_star_data(sample, star_index, mc_index, base_path)

    # Placeholder for echelle results (computed only if enabled).
    echelle_results = None

    # ----------------------------------------------------------------------
    # 3. Production of oscillation-related diagnostic plots
    # ----------------------------------------------------------------------
    if generate_oscillation:
        # Plot the full oscillation mode spectrum for the MC realisation.
        plot_oscillation_modes(data, xlim)

    if generate_linewidth:
        # Linewidth as a function of frequency: reveals damping structure.
        plot_linewidth_vs_frequency(data, figsize)

    if generate_amplitude:
        # Plot amplitudes split by ℓ=0,1,2,3 for quick inspection.
        plot_amplitudes_by_degree(data)

    # ----------------------------------------------------------------------
    # 4. Echelle Diagram (requires mode frequencies and Δν)
    # ----------------------------------------------------------------------
    if generate_echelle:
        # Compute and display the echelle diagram.
        # Also extract Δν and Δν₀₂ diagnostics.
        echelle_results = plot_echelle_diagram(data, figsize)

    # ----------------------------------------------------------------------
    # 5. Synthetic time-series generation (RV or photometry)
    # ----------------------------------------------------------------------
    if generate_time_series:
        # Loads the AADG3 config and produces a synthetic time series
        # using the specific MC realisation.
        plot_synthetic_time_series(sample, star_index, mc_index, base_path)

    # ----------------------------------------------------------------------
    # 6. Load and plot from stored NPZ simulation output
    # ----------------------------------------------------------------------
    if generate_from_npz:
        # Load precomputed power spectrum or time-series products
        # stored in a .npz archive.
        star_ID_f = star_name.replace(' ', '_')
        star_folder = os.path.join(base_path, star_ID_f)
        npz_path = os.path.join(star_folder, f"{star_name}.npz")
        plot_std_vs_ints(npz_path)

    # ----------------------------------------------------------------------
    # 7. Return structured results for programmatic use
    # ----------------------------------------------------------------------
    return {
        "data": data,
        "echelle_results": echelle_results
    }

#--------------------------------#

def get_rms_for_time(npz_path, exposure_time_min=None, fontsize=16):
    """
    Highlighting the RMS and ±1σ at a given exposure time with the option of plotting.
    
    Parameters
    ----------
    npz_path : str
        Path to .npz file containing RMS data.
    exposure_time_min : float, optional
        Exposure time (minutes) at which to overlay RMS and ±1σ.
    fontsize : int
        Font size for plot labels and title.
    """

    # ----------------------------------------------------------------------
    # 1. Load data and convert times to minutes
    # ----------------------------------------------------------------------
    data = np.load(npz_path)
    ints = data['ints'] / 60
    mean_std_dev = data['mean_std_dev']
    std_std_dev = data['std_std_dev']
    star_name = os.path.splitext(os.path.basename(npz_path))[0]

    # ----------------------------------------------------------------------
    # 2. Plot mean RMS and ±1σ band
    # ----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.fill_between(ints, mean_std_dev - std_std_dev, mean_std_dev + std_std_dev,
                     color='purple', alpha=0.2, label='±1σ spread')
    plt.plot(ints, mean_std_dev, color='black', lw=1, label='Mean realization')

    # ----------------------------------------------------------------------
    # 3. Optionally overlay RMS at specific exposure time
    # ----------------------------------------------------------------------
    if exposure_time_min is not None:
        if exposure_time_min < ints.min() or exposure_time_min > ints.max():
            raise ValueError(f"Exposure time {exposure_time_min:.2f} min is outside range ({ints.min():.2f}–{ints.max():.2f} min).")

        mean_interp = interp1d(ints, mean_std_dev, kind='linear')
        std_interp = interp1d(ints, std_std_dev, kind='linear')

        rms = float(mean_interp(exposure_time_min))
        rms_err = float(std_interp(exposure_time_min))

        plt.errorbar(
            exposure_time_min, rms,
            yerr=rms_err,
            fmt='o', color='green', capsize=5,
            label=f'P-mode RMS @ {exposure_time_min:.1f} min = {rms:.2f} ± {rms_err:.2f} m/s'
        )
        plt.axvline(exposure_time_min, color='crimson', linestyle='--',
                    label=f'Target exposure time = {exposure_time_min:.2f} min')

    # ----------------------------------------------------------------------
    # 4. Finalize plot
    # ----------------------------------------------------------------------
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposure time (minutes)', fontsize=fontsize)
    plt.ylabel('RMS Amplitude (m/s)', fontsize=fontsize)
    plt.title(f'{star_name} – P-mode RMS at {exposure_time_min:.2f} min(s)', fontsize=fontsize+2)
    plt.legend()
    plt.tight_layout()
    plt.show()

#--------------------------------#

def get_time_for_rms(npz_path, target_rms, show_plot=True, figsize=(10, 6), fontsize=16):
    """
    Determine integration times where the mean RMS crosses a target value and 
    show ±1σ intervals around those crossings.
    
    Parameters
    ----------
    npz_path : str
        Path to .npz file with RMS vs. integration time data.
    target_rms : float
        Target RMS value (m/s) to locate.
    show_plot : bool
        If True, plot RMS curves and crossings.
    figsize : tuple
        Figure size for the plot.
    fontsize : int
        Font size for labels and title.
    
    Returns
    -------
    crossings : np.ndarray
        Integration times (minutes) where mean RMS equals target RMS.
    time_band_errors : np.ndarray
        2xN array of horizontal ±1σ error bars for each crossing.
    """

    # ----------------------------------------------------------------------
    # 1. Load RMS data and convert times to minutes
    # ----------------------------------------------------------------------
    data = np.load(npz_path)
    ints = data['ints'] / 60
    mean_std_dev = data['mean_std_dev']
    std_std_dev = data['std_std_dev']

    # ----------------------------------------------------------------------
    # 2. Interpolate curves for smooth evaluation
    # ----------------------------------------------------------------------
    mean_interp = interp1d(ints, mean_std_dev, kind='cubic', bounds_error=False, fill_value='extrapolate')
    std_interp = interp1d(ints, std_std_dev, kind='cubic', bounds_error=False, fill_value='extrapolate')
    time_fine = np.logspace(np.log10(ints.min()), np.log10(ints.max()), 1000)
    vals = mean_interp(time_fine)
    lower = vals - std_interp(time_fine)
    upper = vals + std_interp(time_fine)

    # ----------------------------------------------------------------------
    # 3. Find mean RMS crossings of target RMS
    # ----------------------------------------------------------------------
    def find_crossings(y, target):
        sign_change = np.diff(np.sign(y - target))
        indices = np.where(sign_change != 0)[0]
        crossings = []
        for i in indices:
            x0, x1 = time_fine[i], time_fine[i + 1]
            y0, y1 = y[i], y[i + 1]
            slope = (y1 - y0) / (x1 - x0)
            cross_x = x0 + (target - y0) / slope
            crossings.append(cross_x)
        return np.array(crossings)

    crossings = find_crossings(vals, target_rms)
    if len(crossings) == 0:
        print("No crossing found for mean RMS curve with target RMS.")
        return None, None

    # ----------------------------------------------------------------------
    # 4. Identify regions where target RMS lies within ±1σ
    # ----------------------------------------------------------------------
    in_band = (lower <= target_rms) & (target_rms <= upper)
    labels, num_features = label(in_band)
    regions = [(time_fine[np.where(labels==i)[0][0]], time_fine[np.where(labels==i)[0][-1]]) 
               for i in range(1, num_features+1)]

    # Determine ±1σ horizontal errors for each crossing
    xerrs = []
    for c in crossings:
        found = False
        for t_min, t_max in regions:
            if t_min <= c <= t_max:
                xerrs.append([[c - t_min], [t_max - c]])
                found = True
                break
        if not found:
            xerrs.append([[0.0], [0.0]])
    time_band_errors = np.hstack(xerrs)

    # ----------------------------------------------------------------------
    # 5. Optional plot
    # ----------------------------------------------------------------------
    if show_plot:
        star_name = os.path.splitext(os.path.basename(npz_path))[0]
        plt.figure(figsize=figsize)
        plt.plot(ints, mean_std_dev, color='black', label='Mean RMS')
        plt.fill_between(ints, mean_std_dev-std_std_dev, mean_std_dev+std_std_dev, color='purple', alpha=0.2, label='±1σ band')
        plt.axhline(target_rms, color='crimson', linestyle='--', label=f'Target RMS = {target_rms:.2f} m/s')
        plt.errorbar(crossings, [target_rms]*len(crossings), xerr=time_band_errors, fmt='o', color='green', ecolor='green', capsize=5, label='Crossing ± uncertainty')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Exposure time (minutes)', fontsize=fontsize)
        plt.ylabel('RMS Amplitude (m/s)', fontsize=fontsize)
        plt.title(f'{star_name} – Exposure times for RMS = {target_rms:.2f} m/s', fontsize=fontsize+2)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return crossings, time_band_errors

#--------------------------------#

