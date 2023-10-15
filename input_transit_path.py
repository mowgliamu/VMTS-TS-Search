# ====================================
# Input parameters for ab initio data
# ====================================

# Path for ab inito data
input_data_dir 	= '/home/p2goel/PhD_Phase_3/Mad_Pes_Evb/mad_pes_final_version/normal_coordinates_new_version/HF_321G/Baker_11'

# Number of minima [energy wells]
nel = 2

# Ab initio data information
min_A 		= 'r_small'
min_B 		= 'p_small'
ts_AB 		= 'ts_final'

# Job options
queue = False
calc_level = 'hf/3-21g'

# Set Reference State 
reference_state = ('TS', 0, 0)	# A Tuple

# Project out Translation and/or Rotation
project_translation 	= True
project_rotation 	= False
add_rotational_energy 	= False

# Select normal coordinates [mass-weighted, dimensionless]
dimensionless = False
scale_window  = False

# Ab initio program 
est_package = 'GAUSSIAN'

#GAUSSIAN
data_minima = {0: ['Minima-A', input_data_dir+'/'+min_A+'.log', input_data_dir+'/'+min_A+'.fchk'],
	       1: ['Minima-B', input_data_dir+'/'+min_B+'.log', input_data_dir+'/'+min_B+'.fchk']}

data_tstate = {0: ['TS',input_data_dir+'/'+ts_AB+'.log', input_data_dir+'/'+ts_AB+'.fchk']}


# Print optional output info [can slow down performance, use only if needed]
verbose = False

