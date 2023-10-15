# -*- coding: utf-8 -*-
#!usr/bin/env python

import re
import os
import sys
import numpy as np
from libspec import path_exists
from subprocess import call, check_call, CalledProcessError, check_output

from x_to_q import *
from alignment import *
from gaussian_interface import *
from physical_constants import *
from process_ab_initio_data import *

from input_cubic_transit import *

# Set print options
np.set_printoptions(suppress=True)


def linear_transit_path(q_a, q_b):

    '''
    q(s) = q_0 + s*q_1
    '''

    q_0 = (q_a + q_b)/2.0
    q_1 = (q_b - q_a)/2.0

    return q_0 + s*q_1


def quadratic_transit_path(s, q_a, q_b, q_ts):

    '''
    q(s)= q_0 + s*q_1 + 0.5*(s**2.0)*q_2
    '''

    q_0 = q_ts
    q_1 = (q_b - q_a)/2.0
    q_2 = q_a + q_b - 2.0*q_ts
    
    return q_0 + s*q_1 + 0.5*(s**2.0)*q_2


def cubic_transit_path(s, q_a, q_b, q_ts, eigvec):

    '''
    q(s)= q_0 + s*q_1 + (s**2.0)*q_2 + (s**3.0)q_3
    '''

    q_0 = q_ts
    q_1 = eigvec
    q_2 = q_a + q_b - 2.0*q_0
    q_3 = 3.0*(q_b - q_a - 2.0*q_1)
    #q_2 = (q_a + q_b - 2.0*q_0)/2.0
    #q_3 = (q_b - q_a - 2.0*q_1)/2.0

    return q_0 + s*q_1 + 0.5*(s**2.0)*q_2 + (1./6.)*(s**3.0)*q_3


def process_grid_point(cycle, s, param_list, q_a, q_b, q_ts, eigvec, path_type):

    '''
    For a given grid point s, do the following:
	- calculate q, transform to cartesian
	- write an input file for single point calculation
	- submit the calculation, either using queue or directly
	- read the log file and fchk file for energy and gradient

    '''

    natom	=	param_list[0]
    masses	=	param_list[1]
    nmode	=	param_list[2]
    X_reference	=	param_list[3]

    # Get Q coordinate
    if path_type == 'cubic':
	q_current = cubic_transit_path(s, q_a, q_b, q_ts, eigvec)
    elif path_type == 'quadratic':
	q_current = quadratic_transit_path(s, q_a, q_b, q_ts)
    else:
	pass

    # Transform to X
    x_current = np.reshape(np.dot(Q_to_X, q_current), (natom, 3)) + X_reference

    # Align X with X_reference
    eck, B = eckart_rotation_general(natom, masses, x_current, X_reference)
    x_current = np.dot(eck, x_current.T).T

    # Write a gaussian input file for SP calculation (from blurb_ts)
    # Create ts_guess.com from blurb with this geometry
    #h = open(min_A + '.com').readlines()
    h = open('blurb_ts').readlines()
    h[0] = '%Chk=grid_'+str(cycle)+'.chk\n'
    h[1] = '# force '+ calc_level + ' symmetry=com units=au\n'
    for i in range(natom):
	current_geometry = h[i+6].strip().split()
	current_geometry[1] = str(x_current[i, 0])
	current_geometry[2] = str(x_current[i, 1])
	current_geometry[3] = str(x_current[i, 2])
	h[i+6] = "  ".join(current_geometry)+"\n"
    
    hwrite = open('grid_'+str(cycle)+'.com', 'w')
    hwrite.writelines(h)
    hwrite.close()

    # Launch a Gaussian calculation
    # Write joball and submit
    write_joball = open('joball_'+str(cycle), 'w')
    write_joball.write('g09 grid_'+str(cycle)+'.com'+'\n')
    write_joball.close()
    check_call('cp submit temp', shell=True)	
    with open('temp', "r") as fin:
	with open('submit_'+str(cycle), "w") as fout:
	    for line in fin:
		fout.write(line.replace('joball', 'joball_'+str(cycle)))
    temp = check_output(["qsub", "submit_"+str(cycle)])
    current_jobid = int(temp.strip().split('.')[0])

    return current_jobid


def driver_process_all_grid(s_grid, param_list, path_params, outfile_name, path_type):

    '''
    '''

    # Parameters
    Q_A    = path_params[0]
    Q_B    = path_params[1]
    Q_TS   = path_params[2]
    eigvec = path_params[3]

    natom = param_list[0]
    # Initialize
    igrid          = len(s_grid)
    all_energies   = np.zeros(igrid)
    all_gradnorms  = np.zeros(igrid)
    all_gradients  = np.zeros((igrid, 3*natom))
    all_geometries = np.zeros((igrid, natom, 3))

    for i in range(igrid):

	# Generate fchk
	call(['formchk', 'grid_'+str(i)+'.chk'])
	# Read log and fchk file
	ener = read_log_gaussian('grid_'+str(i)+'.log')
	natom, anum, masses, geom, grad, fcm = read_fchk_gaussian('grid_'+str(i)+'.fchk')

	# PG, APR 25: Align current geometry with X_reference
	eck, B = eckart_rotation_general(natom, amass, geom, X_reference)
	geom = np.dot(eck, geom.T).T

	# Store data 
	all_energies[i]     = ener
	all_gradients[i,:]  = grad
	all_gradnorms[i]    = np.linalg.norm(grad)
	# PG Apr 25: CHANGED UNITS OF COORDINATES FROM NBOHR TO ANG
	all_geometries[i,:] = geom


    # Write data to a file
    np.savez(outfile_name, energies=all_energies, gradients=all_gradients,\
			geometries=all_geometries, gradnorms=all_gradnorms)

    # Write geometries to a separate text file [can be used for making animation etc]
    gwrite = open(outfile_name+'_geom.xyz', 'w')
    for k in range(igrid):
	gwrite.write(str(natom)+'\n')
	for i in range(natom):
	    gwrite.write("{:4d}".format(anum[i]))
	    for j in range(3):
		gwrite.write("{:20.8f}".format(all_geometries[k,:][i][j]*bohr_to_ang))
	    gwrite.write('\n')
	gwrite.write('\n')
	gwrite.write('\n')
    gwrite.close()


    return


def check_qsub_status(all_id):

    # Check for job completion and continue
    all_status = []
    for i in range(igrid):
	jobstatus = os.system('qstat ' + str(all_id[i]) + '.nlogn')
	all_status.append(jobstatus)

    return all_status


# ====
# MAIN
# ====


if (__name__ == '__main__'):

    # Move to input data directory
    os.chdir(input_data_dir)

    # ===================================
    # First driver: Process abinitio data
    # ===================================

    print '---------------------------------------'
    print 'Step one: Processing ab initio data...'
    print '---------------------------------------'

    #*******************************************************************************************
    all_minima, all_tstate = driver_process_abinitio_data(est_package, data_minima, data_tstate)
    #*******************************************************************************************

    #sys.exit()

    # This has to be fixed. Well, kinda works. GLOBAL VARIABLES. [COMMON]
    natom = all_minima[0][0].natom
    amass = all_minima[0][0].amass
    nmode = all_minima[0][0].nmode

    # Reference State Data
    ref_L_mat = all_tstate[reference_state[1]][reference_state[2]].Lmwc
    X_reference = all_tstate[reference_state[1]][reference_state[2]].eq_geom_cart
    reference_frequencies = all_tstate[reference_state[1]][reference_state[2]].frequencies

    # Parameter List!
    param_list = [natom, amass, nmode, X_reference, reference_frequencies, ref_L_mat]
    
    #sys.exit()

    # ========================================
    # Second driver: Coordinate Transformation
    # ========================================

    print '-----------------------------------'
    print 'Step two: Processing coordinates...'
    print '-----------------------------------'

    #*********************************************************************************************************
    abinit_min, abinit_ts  = driver_process_coordinate_transformation(all_minima, all_tstate, reference_state)
    #*********************************************************************************************************


    Q_A = abinit_min[0][0].getPosition()
    Q_B = abinit_min[1][0].getPosition()
    Q_TS= abinit_ts[0][0].getPosition()

    #rc_eigvec = np.dot(X_to_Q, np.dot(mass_matrix_sqrt_div, rc_eigvec))
    rc_eigvec = np.zeros(nmode)
    rc_eigvec[3] = 1.0

    path_parameters = [Q_A, Q_B, Q_TS, rc_eigvec]

    # ============================================
    # X->Q and Q->X Transformation (Write to file)
    # ============================================

    Q_to_X, X_to_Q = get_x_to_q_transformation_matrix(param_list, True)


    # Create " the s grid " and calculate data for  all points [Can be parallelized]
    s_grid = np.arange(-1, 1.05, 0.05)
    igrid = len(s_grid)

    # Process Quadratic Data
    all_id = []
    
    # Submit jobs and collect jobid
    for i in range(igrid):
    	jobid = process_grid_point(i, s_grid[i], param_list, Q_A, Q_B, Q_TS, rc_eigvec, 'quadratic')
	all_id.append(jobid)
	call(["sleep", "1.0"])

    # Check status
    all_status = check_qsub_status(all_id)
    while np.count_nonzero(all_status) != igrid:
	print 'Quadratic grid still running, will sleep some more'
	os.system('sleep 20.0')
	all_status = check_qsub_status(all_id)

    print
    print 'Quadratic grid completed. Will process data points.'
    print

    # Quadratic Path
    driver_process_all_grid(s_grid, param_list, path_parameters, 'quadratic_transit_data', 'quadratic')

    # Remove files
    os.system('rm joball_* submit_* grid_*')

    # Process Cubic Data
    all_id = []
    
    # Submit jobs and collect jobid
    for i in range(igrid):
    	jobid = process_grid_point(i, s_grid[i], param_list, Q_A, Q_B, Q_TS, rc_eigvec, 'cubic')
	all_id.append(jobid)
	call(["sleep", "1.0"])

    # Check status
    all_status = check_qsub_status(all_id)
    while np.count_nonzero(all_status) != igrid:
	print 'Cubic grid still running, will sleep some more'
	os.system('sleep 20.0')
	all_status = check_qsub_status(all_id)

    print
    print 'Cubic grid completed. Will process data points.'
    print

    # Cubic Path
    driver_process_all_grid(s_grid, param_list, path_parameters, 'cubic_transit_data', 'cubic')

    # Remove files
    os.system('rm joball_* submit_* grid_*')

    # Move back to current directory
    os.system('cd -')
else:
    pass



