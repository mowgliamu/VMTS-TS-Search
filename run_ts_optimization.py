# -*- coding: utf-8 -*-
#!usr/bin/env python
"""
A program to drive calculation for ts search using VIBRON.

Autthor: Prateek Goel
E-mail: p2goel@uwaterloo.ca
"""

import re
import os
import sys
import math
import numpy as np
from libspec import path_exists
from subprocess import call, check_call, CalledProcessError

from alignment import *
from gaussian_interface import *
from physical_constants import *
from input_vmts import *

np.set_printoptions(suppress=True)

# Output Directory
name_dir = sys.argv[1]
os.chdir(name_dir)

# Clean before starting a new calculation
os.system('rm Vibron* out* Gau-* Pre* summary_ts_search.txt')

# Redirect Output to File
fd = os.open('output_ts_search_script.txt', os.O_RDWR|os.O_CREAT)
sys.stdout = os.fdopen(fd, 'w', 0)

########################
# Steps of calculation #
########################

#1. Read initial set of ab initio data [Minima, and guess TS, if provided]
#   If ts_guess not provided, will start from XA + XB / 2 (if asked to do so)

#2. Launch vibronic model construction with this initial set of data.

#3. Launch a vibron calculation with this vibronic model to perform
#   geometry optimization - obtain first (well, next) guess for TS.

#4. Read output from VIBRON - new geometry and hessian - check if new
#   geometry is same as the provided input. If yes, TS has been found.
#   [Verify that the gradient is zero and hessian has one -ve eigenvalue]

#5. If not converged, launch a (single point) Gaussian calculation at this
#   new geometry - get new Energy and Gradeint [Hessian, if asked for].
#   [You should launch a Gaussian calculation anyway for step 4 - to check
#    convergence]

#6. Go to Step #2 - minima information does not change, only TS gets updated.


########################
# 	SUBROUTINES    #
########################


def create_starting_guess_ts(reactant, product):

    '''
    Read Reactant and Product Optimized Geometry from respective
    Formchk files. Create two Gaussian input files:

    - One Single Point Calculation with (X_A + X_B )/2 as guess
    - Another for QST2 Transition State Optimization

    '''

    if read_ts_guess_from_file:
	print
	print 'Taking first guess from given input file'
	print

	if os.path.isfile(ts_guess_start_filename+'.fchk'):
	    call('cp '+ts_guess_start_filename+'.fchk ts_guess.fchk' , shell=True)
	else:
	    raise IOError('Formchk file not found for initial guess as requested.')

	if os.path.isfile(ts_guess_start_filename+'.log'):
	    call('cp '+ts_guess_start_filename+'.log ts_guess.log' , shell=True)
	else:
	    raise IOError('Log file not found for initial guess as requested.')

	# Read formchk
	natom, anums, masses, X_ts_guess, evals, fcm = read_fchk_gaussian('ts_guess.fchk')

    else:
	print
	print 'Generating first ts guess as XA + XB / 2'
	print

	# Reactant data
	natom, anums, masses, X_A, evals, force_constant_matrix = read_fchk_gaussian(reactant)

	# Product data
	natom, anums, masses, X_B, evals, force_constant_matrix = read_fchk_gaussian(product)

	# Rotate X_B wrt X_A
	print 'Align Minima-B with Minima-A to calculate XA + XB / 2'
	eck, B = eckart_rotation_general(natom, masses, X_B, X_A)
	X_B = np.dot(eck, X_B.T).T

	# Get (X_A + X_B)/2
	X_ts_guess = (X_A + X_B)/2.0

	# PG, MAY 25: SEEMS DISTANCE CRITEIRA IS WRONG DUE TO UNITS!

	#dis_mat_guess = get_distance_matrix(natom, X_ts_guess)

	#if np.any(dis_mat_guess < 0.3):
	#    print
	#    print 'Small interatomic distances found for A+B/2 guess.'
	#    print 'Will try A/3 + 2B/3'
	#    print
	#    X_ts_guess = (2.0*X_A/3.0) + (2.0*X_B/3.0)
	#else:
	#    pass

	# Create ts_guess.com from blurb with this geometry
	h = open('blurb_ts').readlines()
	h[1] = '# '+ small + ' freq symmetry=com units=au\n'
	for i in range(natom):
	    current_geometry = h[i+6].strip().split()
	    current_geometry[1] = str(X_ts_guess[i, 0])
	    current_geometry[2] = str(X_ts_guess[i, 1])
	    current_geometry[3] = str(X_ts_guess[i, 2])
	    h[i+6] = "  ".join(current_geometry)+"\n"

	hwrite = open('ts_guess.com', 'w')
	hwrite.writelines(h)
	hwrite.close()

	# Launch a Gaussian calculation
	try:
	    call('rm ts_guess.chk', shell=True)
	except:
	    pass
	call('g09 ts_guess.com', shell=True)
	call('formchk ts_guess.chk', shell=True)

    return natom, X_ts_guess


def call_construction_of_vibronic_model(cycle, hessian_update_index):

    '''
    '''

    # NOTE: CREATE JOBALL FOR PYTHON AND VIBRON

    print
    print 'Working on Vibronic Model Optimization...'
    print
    print '@call_construction_of_vibronic_model: hessian_update =', hessian_update_index
    print

    # Call construction of vibronic model with this guess
    try:
	if hessian_update_index == 0:
	    check_call('python ../construct_vibronic_model.py 0', shell=True)
	elif hessian_update_index == 1:
	    check_call('python ../construct_vibronic_model.py 1 ts_hess_calc_1.fchk', shell=True)
	elif hessian_update_index == 1.5:
	    check_call('python ../construct_vibronic_model.py 3', shell=True)
	elif hessian_update_index == 2:
	    check_call('python ../construct_vibronic_model.py 2 Vibron_TS_info', shell=True)
	elif hessian_update_index == 5:
	    check_call('python ../construct_vibronic_model.py 5 Bofill_Hessian', shell=True)
	else:
	    pass
    except CalledProcessError as grepexc:
    	print "exit_code", grepexc.returncode
    	sys.exit('Something wrong with vibronic model optimization. Exiting...')

    print
    print 'Vibronic Model Optimization Completed Successfully!'
    print 'Will launch a VIBRON calculation.'
    print

    # Remove Vibron_TS_info, if present
    if os.path.isfile('Vibron_TS_info'):
	call(["rm", "Vibron_TS_info"])
    else:
	print 'No Vibron_TS_info file found'

    # Launch a VIBRON geometry optimization calculation
    filename = "{:02}".format(cycle)
    os.system(xvibron_path+'xvibron_amanda < ZMAT  > out_'+filename)
    #os.system(xvibron_path+'xvibron_amanda_new < ZMAT  > out_'+filename)
    #os.system(xvibron_path+'xvibron_prateek < ZMAT  > out_'+filename)

    print
    print 'VIBRON done. Will read output data.'
    print

    # Read output from VIBRON, do whatever transformations are needed
    # a) from normal modes to cartesian b) put it in it's own Eckart frame

    # Remove Vibron_TS_info, if present
    if os.path.isfile('Vibron_TS_info'):
        fread = open('Vibron_TS_info', 'r')
    else:
	print 'No Vibron_TS_info file found'
	sys.exit(1)

    # Read Coordinates first
    f = fread.readlines()
    fread.close()
    for line in f:
	if 'New coordinates' in line:
	    coord_index = f.index(line)
	else:
	    pass

    nmode = int(f[coord_index+1].strip().split()[0])
    q_ts_current = np.zeros(nmode)
    for i in range(nmode):
	q_ts_current[i] = float(f[i+2+coord_index].strip().split()[0])


    # =======================================
    # NORMAL MODE TO CARTESIAN TRANSFORMATION
    # =======================================

    g = open('transform_cartesian_normal').readlines()
    natom = int(g[0].strip().split()[0])
    amass = np.array([float(k) for k in g[1].strip().split()])

    # Reference Geometry
    for line in g:
	if 'Cartesian Reference Geometry' in line:
	    ref_geom_index = g.index(line)
	else:
	    pass

    ref_cart = np.zeros((natom, 3))
    for i in range(natom):
	ref_cart[i:,] = np.array([float(k) for k in g[i+1+ref_geom_index].strip().split()])


    # Reference Normal Modes
    for line in g:
	if 'Reference Normal Modes' in line:
	    ref_modes_index = g.index(line)
	else:
	    pass

    L_ref_mat = np.zeros((3*natom, nmode))
    for i in range(3*natom):
	L_ref_mat[i:,] = np.array([float(k) for k in g[i+1+ref_modes_index].strip().split()])


    # Reference Frequencies
    for line in g:
	if 'Reference Frequencies' in line:
	    ref_freq_index = g.index(line)
	else:
	    pass

    ref_freq = np.zeros(nmode)
    ref_freq = np.array([float(k) for k in g[1+ref_freq_index].strip().split()])


    #---------------------------------------
    #DIMENSIONLESS  X <--> Q TRANSFORMATIONS
    #---------------------------------------

    X_to_Q = np.zeros(np.shape(L_ref_mat)).T
    for alpha in range(nmode):
	for i in range(natom):
	    for j in range(3):
		k = 3*i + j
		if dimensionless:
		    X_to_Q[alpha][k] = (fred*np.sqrt(ref_freq[alpha])*np.sqrt(amass[i]))*L_ref_mat.T[alpha][k]
		else:
		    X_to_Q[alpha][k] = np.sqrt(amass[i])*L_ref_mat.T[alpha][k]

    Q_to_X = np.zeros(np.shape(L_ref_mat))
    for alpha in range(nmode):
	for i in range(natom):
	    for j in range(3):
		k = 3*i + j
		if dimensionless:
		    Q_to_X[k][alpha] = L_ref_mat[k][alpha]/(fred*np.sqrt(ref_freq[alpha])*np.sqrt(amass[i]))
		else:
		    Q_to_X[k][alpha] = L_ref_mat[k][alpha]/(np.sqrt(amass[i]))


    # Transform q_ts_current to X
    x_ts_current = np.reshape(np.dot(Q_to_X, q_ts_current), (natom, 3)) + ref_cart

    return q_ts_current, x_ts_current, L_ref_mat


def launch_energy_gradient_calculation(x_ts_current):

    '''
    Force Constant Matrix (FCM) will NOT be calculated.
    '''

    global natom

    # Update geometry using Blurb
    h = open('blurb_ts').readlines()
    for i in range(natom):
	current_geometry = h[i+6].strip().split()
	current_geometry[1] = str(x_ts_current[i, 0])
	current_geometry[2] = str(x_ts_current[i, 1])
	current_geometry[3] = str(x_ts_current[i, 2])
	h[i+6] = "  ".join(current_geometry)+"\n"

    # Write file for large calc ("freq" excluded, "force" included)
    h[1] = '# force ' + full_gradient +' symmetry=com units=au\n'
    hwrite = open('ts_guess.com', 'w')
    hwrite.writelines(h)
    hwrite.close()

    # Run Gaussian on new ts_guess
    if os.path.isfile('ts_guess.chk'):
	call(["rm", "ts_guess.chk"])
    else:
	pass
    call(["g09", "ts_guess.com"])
    call(["formchk", "ts_guess.chk"])

    # Read new log and checkpoint file
    E0 = read_log_gaussian("ts_guess.log")
    natom, anums, masses, coords, evals, force_constant_matrix = read_fchk_gaussian("ts_guess.fchk")

    return anums, masses, E0, evals, force_constant_matrix


def launch_full_hessian_calculation(hessian_level, x_ts_current):

    '''
    A Gaussian calculation returning full FCM, in addition to E/g.
    '''

    global natom

    # Update geometry using Blurb
    h = open('blurb_ts').readlines()
    for i in range(natom):
	current_geometry = h[i+6].strip().split()
	current_geometry[1] = str(x_ts_current[i, 0])
	current_geometry[2] = str(x_ts_current[i, 1])
	current_geometry[3] = str(x_ts_current[i, 2])
	h[i+6] = "  ".join(current_geometry)+"\n"

    # Write file for large calc ("freq" included))
    h[1] = '# freq ' + hessian_level +' symmetry=com units=au\n'
    hwrite = open('ts_guess.com', 'w')
    hwrite.writelines(h)
    hwrite.close()

    # Run Gaussian on new ts_guess
    if os.path.isfile('ts_guess.chk'):
	call(["rm", "ts_guess.chk"])
    else:
	pass
    call(["g09", "ts_guess.com"])
    call(["formchk", "ts_guess.chk"])

    # Read new log and checkpoint file
    E0 = read_log_gaussian("ts_guess.log")
    natom, anums, masses, coords, evals, force_constant_matrix = read_fchk_gaussian("ts_guess.fchk")

    return anums, masses, E0, evals, force_constant_matrix


def launch_full_hessian_lower_calculation(hessian_level, x_ts_current):

    '''
    A Gaussian calculation returning full FCM, in addition to E/g.
    '''

    global natom

    # Update geometry using Blurb
    h = open('blurb_ts').readlines()
    for i in range(natom):
	current_geometry = h[i+6].strip().split()
	current_geometry[1] = str(x_ts_current[i, 0])
	current_geometry[2] = str(x_ts_current[i, 1])
	current_geometry[3] = str(x_ts_current[i, 2])
	h[i+6] = "  ".join(current_geometry)+"\n"

    # Write file for large calc ("freq" included))
    # Write file for small calc ("freq" included)
    h[0] = h[0].replace('ts_guess', 'ts_hess_calc_1')
    h[1] = '# ' + hessian_level + ' freq symmetry=com units=au\n'
    gwrite = open('ts_hess_calc_1.com', 'w')
    gwrite.writelines(h)
    gwrite.close()

    # Run Gaussian on new ts_guess
    if os.path.isfile('ts_hess_calc_1.chk'):
	call(["rm", "ts_hess_calc_1.chk"])
    else:
	pass
    call(["g09", "ts_hess_calc_1.com"])
    call(["formchk", "ts_hess_calc_1.chk"])

    # Read new log and checkpoint file
    natom, anums, masses, coords, evals, force_constant_matrix = read_fchk_gaussian("ts_hess_calc_1.fchk")

    return force_constant_matrix


def vibronic_model_hessian_update_scheme(natom, x_ts_current):

    '''
    '''

    # Energy and Gradient
    anums, masses, E0, evals, foo = launch_energy_gradient_calculation(x_ts_current)

    # Read VIBRON_TS_INFO

    if os.path.isfile('Vibron_TS_info'):
	fread = open('Vibron_TS_info', 'r')
    else:
	print 'No Vibron_TS_info file found'
	sys.exit(1)

    # Read Coordinates first
    f = fread.readlines()
    fread.close()

    for line in f:
	if 'New coordinates' in line:
	    coord_index = f.index(line)
	else:
	    pass

    nmode = int(f[coord_index+1].strip().split()[0])
    force_constant_matrix = np.zeros((nmode, nmode))
    for line in f:
	if 'Hessian' in line:
	    hess_index = f.index(line)
	else:
	    pass

    hess_cols   = int(f[hess_index+1].strip().split()[2])
    hess_blocks = int(math.ceil(float(nmode)/float(hess_cols)))

    last_block  = nmode % hess_cols

    m = 0
    start_index = hess_index + 2
    for r in range(hess_blocks):

	if r == hess_blocks-1 and last_block != 0:
	    for i in range(nmode):
		force_constant_matrix[i,:][m:m+last_block] = np.array([float(k) for k in f[i+start_index].strip().split()])
	else:
	    for i in range(nmode):
		force_constant_matrix[i,:][m:m+4] = np.array([float(k) for k in f[i+start_index].strip().split()])
	    m = m + 4

	start_index = start_index + nmode + 1


    return anums, masses, E0, evals, force_constant_matrix


def bofill_hessian_update_scheme(natom, x_ts_current, data_previous):

    '''
    A different hessian update procedure.
    '''

    # Energy and Gradient
    anums, masses, E0, evals, foo = launch_energy_gradient_calculation(x_ts_current)

    # ===================
    # HESSIAN CALCULATION
    # ===================

    # Old Data
    x_old = data_previous[0]
    g_old = data_previous[1]
    h_old = data_previous[2]

    # Quantities needed for Bofill Hessian Update
    delta_x 	= (x_ts_current - x_old).flatten()
    delta_g 	= evals - g_old
    g_Hx 	= delta_g - np.dot(h_old, delta_x)
    phi_num	= (np.dot(g_Hx.T, delta_x))**2.0
    phi_den	=  ((np.linalg.norm(g_Hx))**2.0)*(np.linalg.norm(delta_x)**2.0)
    phi 	= phi_num / phi_den

    print
    print '@Bofill_Update: phi', phi
    print

    # SR1 Update
    update_SR1 = np.outer(g_Hx, g_Hx.T)/np.dot(g_Hx.T, delta_x)

    print
    print 'Check if SR1 Hessian is Symmetric'
    print np.allclose(update_SR1, update_SR1.T, atol=1e-8)
    print

    # PSB Update
    update_psb_term_1 = (np.outer(g_Hx, delta_x.T) + np.outer(delta_x, g_Hx.T))/(np.dot(delta_x.T, delta_x))
    #update_psb_term_2 = (np.outer(delta_x.T, g_Hx))*(np.outer(delta_x, delta_x.T))/((np.dot(delta_x.T, delta_x))**2.0)
    update_psb_term_2 = (np.dot(delta_x.T, g_Hx))*(np.outer(delta_x, delta_x.T))/((np.dot(delta_x.T, delta_x))**2.0)
    update_PSB	  = update_psb_term_1 - update_psb_term_2

    print
    print 'Check if PSB Hessian is Symmetric'
    print np.allclose(update_PSB, update_PSB.T, atol=1e-8)
    print

    # BFGS Update
    update_bfgs_term_1 = np.outer(delta_g, delta_g.T)/np.dot(delta_g.T, delta_x)
    update_bfgs_term_2 = np.outer(np.dot(h_old, delta_x), np.dot(delta_x.T, h_old))/np.dot(delta_x.T, np.dot(h_old, delta_x))
    update_bfgs 	   = update_bfgs_term_1 - update_bfgs_term_2

    print
    print 'Check if BFGS Hessian is Symmetric'
    print np.allclose(update_bfgs, update_bfgs.T, atol=1e-8)
    print

    # Bofill Update
    bofill_hessian_update = phi*update_SR1 + (1 - phi)*update_PSB

    print
    print 'Check if Bofill Hessian is Symmetric'
    print np.allclose(bofill_hessian_update, bofill_hessian_update.T, atol=1e-8)
    print

    foo_val, foo_vec = np.linalg.eigh(bofill_hessian_update)

    # Farkas - Schlegel Update
    fs_update = np.sqrt(phi)*update_SR1 + (1 - np.sqrt(phi))*update_bfgs

    # Copy to FCM
    force_constant_matrix = h_old + bofill_hessian_update
    #force_constant_matrix = h_old + fs_update
    #force_constant_matrix = h_old + update_PSB

    # WRITE FCM TO FILE FOR USE IN VIBRONIC MODEL CONSTRUCTION: USE NUMPY SAVETXT!
    np.savetxt('Bofill_Hessian', force_constant_matrix)


    return anums, masses, E0, evals, force_constant_matrix


# =======================================
# WRITE SUMMARY TO FILE IN TABULAR FORMAT
# =======================================

def write_summary():

    # Relative Energy
    summary_dict['E(SCF)'] = summary_dict['E(SCF)'] - np.amin(summary_dict['E(SCF)'])

    # Write to file
    fwrite = open('summary_ts_search.txt', 'w')
    fwrite.write("{:<8} {:<20} {:<20} {:<20} {:<20}".format('#Cycle','Energy','Max Grad', 'Max Geom', 'Calc Type')+'\n')
    fwrite.write("{:<8} {:<20} {:<20} {:<20} {:<20}".format('#=====','======','========', '========', '=========')+'\n\n')
    for i in range(ncycle_converged+1):
	fwrite.write("{:<8} {:<20} {:<20} {:<20} {:<20}".format(i+1, summary_dict['E(SCF)'][i], summary_dict['MaxGrad'][i], summary_dict['MaxGeom'][i], summary_dict['Calc Type'][i])+'\n')
    fwrite.close()

    return


########
# MAIN #
########


# Create fchk files: MAKE BETTER
call(["formchk", "r_small.chk"])
call(["formchk", "p_small.chk"])
call(["formchk", "r_large.chk"])
call(["formchk", "p_large.chk"])
call(["formchk", "qst2.chk"])


# Launch first gaussian calculation
global natom
natom, new_cart = create_starting_guess_ts("r_small.fchk", "p_small.fchk")


#sys.exit()


# ================================================================
# PERFORM A PRE-OPTIMIZATION AT 3-21G (OR SIMILAR) LEVEL OF THEORY
# ================================================================

hessian_update = 0

print
print 'Performing pre-optimization (small basis, most likely hf/3-21g)'
print

# Start Iterative Loop for TS search
flag_convergence = False
for i in range(max_iter_small):
    # Call sequence: vibronic model --> vibron --> gaussian
    new_geom, new_cart, normal_modes 	= call_construction_of_vibronic_model(i, hessian_update)
    anum, amass, new_ener, new_grad, new_fcm 	= launch_full_hessian_calculation(small, new_cart)

    # Check Hessian Eigenvalue
    mass_matrix_sqrt_div 	= np.diag(np.repeat(1.0/np.sqrt(amass), 3))
    mw_fcm 		     	= np.dot(mass_matrix_sqrt_div, np.dot(new_fcm, mass_matrix_sqrt_div))
    LTR, projection 		= project_out_translation_rotation(natom, amass, new_cart)
    Imat 			= np.eye(LTR.shape[0])
    llt 			= np.dot(LTR, LTR.T)
    proj_trans_rot_hessian 	= np.dot(Imat - llt, np.dot(mw_fcm, Imat - llt))
    hval, hvec 			= np.linalg.eigh(proj_trans_rot_hessian)

    print
    print 'Cycle, Update, Energy, Max Geom, Max Grad:', i, hessian_update, new_ener, np.amax(abs(new_grad)), np.amax(abs(new_geom))
    print
    print 'Geometry in current cycle (Normal Mode Coordinate)'
    print new_geom
    print
    print 'Gradient in current cycle (Hartree/Bohr)'
    print new_grad
    print
    print 'Eigenvalues of MW Hessian in current cycle (eV)'
    print hval*au_to_ev
    print

    # Check Convergence
    if np.all(abs(new_geom) < err_tol_q_small) and np.all(abs(new_grad) < err_tol_g_small) and i <= max_iter_small:
	print
	print 'Convergence reached for pre-optimization', i, 'cycles'
	print

	if calc_type == 'large':
		print
		print 'Will now switch to large basis calculation.'
		print 'You have asked for following hessian update scheme:', update_scheme
		print
	else:
	    print
	    print 'Did not ask for a large calculation. Exiting after small (pre-optimization) step.'
	    print
	    sys.exit()

	break
    elif i > max_iter_small:
	print
	print 'Max cycles reached for pre-optimization stage, could not converge.'
	print
	break
    else:
	continue


##########################
# LARGE CALCULATION STARTS
##########################

print
print 'Entering LARGE branch. Finding TS at the (desired) higher level of theory'
print

# Get initial ab initio data at the starting geometry
# (At the geometry obtained by pre-optimization process)
if full_gradient == full_hessian:
    anum, amass, new_ener, new_grad, new_fcm 	= launch_full_hessian_calculation(full_hessian, new_cart)
else:
    anum, amass, new_ener, new_grad, foo_fcm 	= launch_energy_gradient_calculation(new_cart)
    new_fcm 					= launch_full_hessian_lower_calculation(full_hessian, new_cart)


# Initialize dictionary for summary of results
summary_dict = {'E(SCF)': np.zeros(max_iter_large),\
		'MaxGrad': np.zeros(max_iter_large),\
		'MaxGeom': np.zeros(max_iter_large),\
		'Calc Type': np.zeros(max_iter_large)}

all_geometries = np.zeros((max_iter_large+1, natom, 3))
all_geometries[0,:] = new_cart

old_data = [new_cart, new_grad, new_fcm]

# Start Iterative Loop for TS search
if full_gradient == full_hessian:
    hessian_update 	  = 1.5
else:
    hessian_update 	  = 1

calc_full_hessian = 1
flag_convergence  = False

for i in range(max_iter_large):

    print
    print '==========================='
    print 'Current iteration cycle', i
    print '==========================='
    print

    # Remove output_call_vibronic_model, if present
    if os.path.isfile('output_call_vibronic_model'):
	call(["rm", "output_call_vibronic_model"])
    else:
	pass

    # ======================================================================
    # PROCESS VIBRONIC MODEL AND LAUNCH GAUSSIAN: THE MAIN WORK HAPPENS HERE
    # ======================================================================

    previous_geom = new_geom

    # Construct Vibronic Model and find TS on the model (External call to VIBRON)
    new_geom, new_cart, normal_modes 	= call_construction_of_vibronic_model(i, hessian_update)

    # Re-launch Gaussian to obtain ab initio data at the new geometry found by VIBRON.
    # NOTE: Do NOT calculate full hessian again. Use update scheme as asked by the user.

    # But check if difference in geometry is greater than threshold.
    # If YES, well, launch full hessian calculation again. If NOT, use update.

    if np.any(abs(new_geom) - abs(previous_geom)) > geometry_threshold:
	print
	print 'Difference in geometry between two consecutive cycles is too large.'
	print 'Will calculate full Hessian for the next step'
	print
	if full_gradient == full_hessian:
	    anum, amass, new_ener, new_grad, new_fcm 	= launch_full_hessian_calculation(full_hessian, new_cart)
	    calc_full_hessian += 1
	    hessian_update = 1.5
	else:
	    anum, amass, new_ener, new_grad, foo_fcm 	= launch_energy_gradient_calculation(new_cart)
	    new_fcm 					= launch_full_hessian_lower_calculation(full_hessian, new_cart)
	    calc_full_hessian += 1
	    hessian_update = 1
    else:
	if update_scheme == 'Bofill':
	    anum, amass, new_ener, new_grad, new_fcm = bofill_hessian_update_scheme(natom, new_cart, old_data)
	    hessian_update = 5
	elif update_scheme == 'Vibronic':
	    anum, amass, new_ener, new_grad, new_fcm = vibronic_model_hessian_update_scheme(natom, new_cart)
	    hessian_update = 2
	else:
	    pass

    # Update old data for next cycle
    old_data[0] = np.copy(new_cart)
    old_data[1] = np.copy(new_grad)
    old_data[2] = np.copy(new_fcm)

    # =========================
    # CHECK HESSIAN EIGENVALUES
    # =========================

    if hessian_update == 2:
	hessian_new, Lmwc_old  	= read_vibronic_model_hessian('Vibron_TS_info')
	hessian_new_mwc   	= np.dot(Lmwc_old, np.dot(hessian_new, Lmwc_old.T))
	LTR, projection 	= project_out_translation_rotation(natom, amass, new_cart)
	Imat 			= np.eye(LTR.shape[0])
	llt 			= np.dot(LTR, LTR.T)
	proj_trans_rot_hessian 	= np.dot(Imat - llt, np.dot(hessian_new_mwc, Imat - llt))
	hval, hvec 		= np.linalg.eigh(proj_trans_rot_hessian)
    else:
	mass_matrix_sqrt_div 	= np.diag(np.repeat(1.0/np.sqrt(amass), 3))
	mw_fcm 		     	= np.dot(mass_matrix_sqrt_div, np.dot(new_fcm, mass_matrix_sqrt_div))
	LTR, projection 	= project_out_translation_rotation(natom, amass, new_cart)
	Imat 			= np.eye(LTR.shape[0])
	llt 			= np.dot(LTR, LTR.T)
	proj_trans_rot_hessian 	= np.dot(Imat - llt, np.dot(mw_fcm, Imat - llt))
	hval, hvec 		= np.linalg.eigh(proj_trans_rot_hessian)

    print
    print 'Cycle, Update, Energy, Max Geom, Max Grad:', i, hessian_update, new_ener, np.amax(abs(new_grad)), np.amax(abs(new_geom))
    print
    print 'Geometry in current cycle (Normal Mode Coordinate)'
    print new_geom
    print
    print 'Gradient in current cycle (Hartree/Bohr)'
    print new_grad
    print
    print 'Eigenvalues of MW Hessian in current cycle (eV)'

    if hessian_update == 2:
	print hval
    else:
	print hval*au_to_ev
    print

    # For Bofill Update, check if hessian has more than one negative eigenvalue
    # If yes, exit with an error message [OR WARNING]
    if hessian_update == 5:
	all_index_0 = np.where(hval < -1e-6)[0]
	if len(all_index_0) != 1:
	    print
	    print "WARNING: More than one -ve eigval in Hessian for Bofill Update."
	    print
	else:
	    pass
    else:
	pass

    # STORE DATA
    all_geometries[i+1,:] 		= new_cart
    summary_dict['E(SCF)'][i] 		= new_ener
    summary_dict['MaxGrad'][i] 		= np.amax(abs(new_grad))
    summary_dict['MaxGeom'][i] 		= np.amax(abs(new_geom))
    summary_dict['Calc Type'][i] 	= hessian_update

    # ==================================================
    # CHECK CONVERGENCE AND CHANGE VARIABLES ACCORDINGLY
    # ==================================================

    if np.all(abs(new_geom) < err_tol_q_large) and np.all(abs(new_grad) < err_tol_g_large) and i <= max_iter_large-1:
	print
	print 'Convergence reached after', i, 'cycles'
	print
	flag_convergence = True
	ncycle_converged = i
	break
    if np.all(abs(new_geom) > err_tol_q_large) and np.all(abs(new_grad) > err_tol_g_large) and i == max_iter_large-1:
	print
	print 'Max cycles reached for TS search large, could not converge.'
	print
	ncycle_converged = i
	break
    else:
	continue


# =============
# Write summary
# =============

print
print 'Total # of Full Hessian calls:', calc_full_hessian
print

write_summary()


# Write geometries to a separate text file [can be used for making animation etc]
gwrite = open('intermediate_geometries.xyz', 'w')
for k in range(ncycle_converged+2):
    gwrite.write(str(natom)+'\n')
    for i in range(natom):
	gwrite.write("{:4d}".format(anum[i]))
	for j in range(3):
	    gwrite.write("{:20.8f}".format(all_geometries[k,:][i][j]*bohr_to_ang))
	gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')
gwrite.close()


# Launch a calculation to check final convergence (if asked for)
if flag_convergence and check_convergence:
    natom = np.shape(new_cart)[0]
    h = open('blurb_ts').readlines()
    for i in range(natom):
	current_geometry = h[i+6].strip().split()
	current_geometry[1] = str(new_cart[i, 0])
	current_geometry[2] = str(new_cart[i, 1])
	current_geometry[3] = str(new_cart[i, 2])
	h[i+6] = "  ".join(current_geometry)+"\n"

    h[0] = h[0].replace('ts_guess', 'ts_converged')
    if calc_type == 'small':
	h[1] = '# ' + small +' freq symmetry=com units=au\n'
    elif calc_type == 'large':
	h[1] = '# ' + full_gradient +' freq symmetry=com units=au\n'
    else:
	raise ValueError('Calculation type not understood: can be small or large')

    hwrite = open('ts_converged.com', 'w')
    hwrite.writelines(h)
    hwrite.close()

    # Run Gaussian on new ts_guess
    if os.path.isfile('ts_converged.chk'):
	call(["rm", "ts_converged.chk"])
    else:
	pass
    call(["g09", "ts_converged.com"])
    call(["sleep", "5.0"])
    call(["formchk", "ts_converged.chk"])

    # Read new checkpoint file
    natom, anums, amass, coords, evals, force_constant_matrix = read_fchk_gaussian("ts_converged.fchk")

    # ===========================
    # FINAL PROCESSING OF RESULTS
    # ===========================

    # Get the matrix of atomic masses
    mass_matrix_sqrt_div = np.diag(np.repeat(1.0/np.sqrt(amass), 3))

    # Mass-weight the force constant matrix
    mw_fcm = np.dot(mass_matrix_sqrt_div, np.dot(force_constant_matrix, mass_matrix_sqrt_div))

    # Project out translation and rotation from the mass-weighted FCM
    LTR, projection = project_out_translation_rotation(natom, amass, coords)

    # Identity Matrix
    Imat = np.eye(LTR.shape[0])
    qqp = np.dot(LTR, LTR.T)

    # Project the force constant matrix (1 - P)*H*(1 - P)
    proj_hessian =  np.dot(Imat - qqp, np.dot(mw_fcm, Imat - qqp))

    # Diagonalize (Eigenvalues are sorted)
    hval, hvec = np.linalg.eigh(proj_hessian)

    # E(SCF)
    E0 = read_log_gaussian("ts_converged.log")

    print
    print 'Final SCF Energy'
    print E0
    print
    print 'All eigenvalues of hessian for located transition state'
    print np.sign(hval)*np.sqrt(abs(hval))*hfreq_cm
    print

    if perform_irc_if_converged:
	# Prepare an IRC input file
	h[0] = h[0].replace('ts_converged', 'irc')
	h[1] = '# irc=(maxpoints=300,stepsize=3,recalc=3,calcfc,maxcycle=300) hf/3-21g symmetry=com units=au\n'

	for i in range(natom):
	    current_geometry = h[i+6].strip().split()
	    current_geometry[1] = str(coords[i, 0])
	    current_geometry[2] = str(coords[i, 1])
	    current_geometry[3] = str(coords[i, 2])
	    h[i+6] = "  ".join(current_geometry)+"\n"

	gwrite = open('irc.com', 'w')
	gwrite.writelines(h)
	gwrite.close()

	call(["g09", "irc.com"])
	call(["formchk", "irc.chk"])
    else:
	pass

elif flag_convergence and not launch_gaussian_opt:
    pass
else:
    sys.exit('Convergence not reached after max cylces for ts search')

print
print 'Automated TS search using MAD-PES completed successfully.'
print

os.chdir('../')
