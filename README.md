Introduction
#----------------#

When searching for Earth-like planets around F, G, and K-type stars, pressure-mode (P-mode) oscillations, caused by global acoustic waves in the stellar interior, induce RV variations on the m/s scale, potentially masking Earth-like planetary signals (cm/s). Mitigating these oscillations is crucial for the RV follow-up of potentially habitable exoplanets can be done by tuning exposure times to stellar parameters.

For Sun-like stars, P-mode oscillations occur over a narrow frequency range. They can, therefore, be averaged out to reduce the error in the RV measurements. Chaplin et al. (2019), who used the stellar parameters to estimate the RV amplitude, observed that the P-mode signal does not monotonically decrease with time, further emphasising the need to determine the optimal exposure time to remove the P-mode noise.

Exposure Time Evaluation for Mitigation of P-mode Oscillations (ExTEMPO)
#----------------#

ExTEMPO is a tool which can be used to develop a p-mode mitigation strategy by optimising the exposure times of radial velocity observations. It uses the stellar parameters such as mass, radius, effective temperature, log luminosity, nu_max and delta_nu to determine the P-mode rms amplitude at different exposure times for solar-like stars (F, G and K stars). ExTEMPO utilises stellar scaling relations and contributions taken from Kjeldsen & Bedding (1994), White et al. (2011), Mosser et al. (2013), Ball et al. (2018) and Nielson et al. (2021), to model the PSD of the p-mode oscillations. The mode data is then fed to the AsteroFLAG Artificial Dataset Generator 3 (AADG3), which simulates a radial velocity time series. The time series is then integrated to obtain the rms amplitude at different exposure times. Monte Carlo (MC) sampling of the input parameters is used to generate multiple realisations of the target star. Each realisation is processed as previously described, producing a distribution of rms amplitude curves from which a one-sigma error is obtained.

ExTEMPO is in active development, so please report any issues/suggestions on GitHub (https://github.com/BRajkumar041992). The first version is in the form of a Jupyter notebook (ExTEMPO_v1.0.ipynb) for easy access. 

To run ExTEMPO, first ensure the following packages are preinstalled.

os
time
hashlib
subprocess
multiprocessing
concurrent.futures
numpy
pandas
matplotlib
scipy
glob
re
ADDG3 (https://warrickball.gitlab.io/AADG3/index.html#)

ExTEMPO also requires a CSV file containing target stars and their stellar and asteroseismic parameters. The file should contain columns such as primary_name, Teff(K), Teff_err(K), Mass, Mass-err, Mass+err, Radius, Radius-err, Radius+err, logLuminosity, logLuminosity-err, logLuminosity+err, deltanu, deltanu-err, deltanu+err, numax, numax-err, and numax+err. A sample file is provided on GitHub.

Open the .ipymb file. The 1st cell imports needed packages. The 2nd cell defines the functions and classes of ExTEMPO. The 3rd cell requires user input. A description of each input is also included. Once inputs have been defined, the 4th cell runs the code, which will generate a folder for the target star, subfolders for each MC realisation and a MC_parameters.csv file containing the sampled parameters for each realisation. Each subfolder will contain:

- a PSD.csv file, which contains the data for the generated power spectral density.
- .con and .in files, which are the generated inputs for ADDG3. The .con file also contains the mode amplitude, frequency and widths, which are used to generate various plots in the 5th cell.
- a .asc file, the output of ADDG3, which contains the simulated time series.
- a .npz file, which contains the rms amplitudes and exposure times for each realisation, the mean rms amplitude plot and its errors.

Cell 5 provides the option to plot:

- The PSD for one MC realisation,
- the mode/line width vs radial order for one MC realisation,
- the mode amplitude vs radial order for one MC realisation,
- the echelle diagram for one MC realisation,
- the synthetic time series generated from ADDG3 for one MC realisation,
- The RMS amplitude plot with each realisation, the mean realisation, and the estimated nu_max.

Each plot can be toggled on or off, and the user can define which MC realisation they would like to plot.

Cell 6 allows the user to define a desired exposure time and target RMS amplitude. Cell 7 determines the RMS amplitude obtained at the defined exposure time and its associated error. It also produces a plot for visual inspection. Cell 8 determines all exposure times and their errors where the RMS amplitude curve crosses the user-defined target RMS amplitude. 


















