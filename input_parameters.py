# ====================================
# Input parameters for ab initio data
# ====================================

# Output Directory
name_dir 	= 'prateek_output'

# Path for ab inito data
input_data_dir 	= '/home/nooijen/Tests/prateek_vmts_g16/A_10/'

redirect_output_to_file = 'output.txt'

# Ab initio program 
est_package = 'GAUSSIAN'
gaussian_version = 'g16'

# Number of minima [energy wells]
nel = 2

# Ab initio data information
min_A 		= 'r_large'
min_B 		= 'p_large'
ts_AB 		= 'ts_converged'

# Window function for transition state(s)
window_flag 	 = 'Gaussian'
window_threshold = 0.05	# alpha window
delta_threshold  = 0.001	# delta window

# PES scan - Analysis of Vibronic Model
analyse_irc 	= True
analyse_qtp 	= False
analyse_ctp 	= False

# IRC Path [after ts search has completed]
irc_path 	= input_data_dir+'/irc.fchk'
irc_all_paths 	= [irc_path]

# Quadratic transit path data
qtp_path 	= input_data_dir+'/quadratic_transit_data.npz'
qtp_all_paths 	= [qtp_path]

# Cubic transit path data
ctp_path 	= input_data_dir+'/cubic_transit_data.npz'
ctp_all_paths 	= [ctp_path]

# Set Reference State 
reference_state = ('TS', 0, 0)	# A Tuple

# Project out Translation and/or Rotation
project_translation 	= True
project_rotation 	= False
add_rotational_energy 	= False

# =========================================
# Universal Rotational Frequency [50 cm^-1]
# =========================================

hfreq_cm = 5140.48
rot_freq_cm = 50.00	# cm-1
rot_freq = (rot_freq_cm/hfreq_cm)**2	# For the Hessian [au / (amu * Bohr^2)]

# Select normal coordinates [mass-weighted, dimensionless]
dimensionless = False
scale_window  = False

# Optimization - not used for ts search [at least for now]
optimize_alpha_beta 	= 'ts_only'
include_grad_norm 	= False

# Beta optimization 
optimize_beta     = False
optimize_two_beta = False
use_data_beta_opt = 'irc'
b_vector_choice   = 'eigvec_ts'

# Threshold for optimization of vibronic model parameters
optimization_threshold = 1e-5

#GAUSSIAN
data_minima = {0: ['Minima-A', input_data_dir+'/'+min_A+'.log', input_data_dir+'/'+min_A+'.fchk'],
	       1: ['Minima-B', input_data_dir+'/'+min_B+'.log', input_data_dir+'/'+min_B+'.fchk']}

data_tstate = {0: ['TS',input_data_dir+'/'+ts_AB+'.log', input_data_dir+'/'+ts_AB+'.fchk']}


# Print optional output info [can slow down performance, use only if needed]
verbose = False




