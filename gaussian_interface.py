# -*- coding: utf-8 -*-
#!usr/bin/env python

# A script to drive calculation for ts search using VIBRON.

import re
import os
import sys
import numpy as np
from libspec import path_exists
from subprocess import call, check_call, CalledProcessError


# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-GAUSSIAN-INTERFACE-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

# ACES fcm output files had neatly organized data, so using line numbers was more relevant.
# Here, things are scattered and messy and all over the place, so using re search is better.

def read_log_gaussian(filename):

    '''
    Read single-point-energy, zero-point-energy
    '''

    E0 = None; ZPE = None

    try:
	f = open(filename, 'r')
    except EnvironmentError:
	print 'Something not right with Gaussian log file.'
	print 'IOError / OSError / WindowsError. Check. Fix. Rerun.'

    line = f.readline()
    while line != '':

	if 'SCF Done:' in line:
	    E0 = float(line.split()[4]) 
	
	# Do NOT read the ZPE from the "E(ZPE)=" line, as this is the scaled version!
	# We will read in the unscaled ZPE and later multiply the scaling factor
	# from the input file

	elif 'Zero-point correction=' in line:
	    ZPE = float(line.split()[2]) 
	elif '\\ZeroPoint=' in line:
	    line = line.strip() + f.readline().strip()
	    start = line.find('\\ZeroPoint=') + 11
	    end = line.find('\\', start)
	    ZPE = float(line[start:end]) 
	else:
	    pass

	# Read the next line in the file
	line = f.readline()

    # Close file when finished
    f.close()
    
    if E0 is not None:
	if ZPE is not None:
	    return E0, ZPE
	else:
	    return E0
    else: raise Exception('Unable to find energy or zpe in Gaussian log file.')


def read_fchk_gaussian(filename, gaussian_version):

    '''
    Read natom, cgeom, anum, amass, force-constant-matrix and also gradients!
    '''

    if gaussian_version == 'g16':
	stuff = re.search(
	    'Atomic numbers\s+I\s+N=\s+(?P<num_atoms>\d+)'
	    '\n\s+(?P<anums>.*?)'
	    'Nuclear charges.*?Current cartesian coordinates.*?\n(?P<coords>.*?)'
	    'Number of symbols in /Mol/'
	    '.*?Real atomic weights.*?\n(?P<masses>.*?)'
	    'Atom fragment info.*?Cartesian Gradient.*?\n(?P<evals>.*?)'
	    'Cartesian Force Constants.*?\n(?P<hess>.*?)'
	    'Nonadiabatic coupling',
	    open(filename, 'r').read(), flags=re.DOTALL)
    elif gaussian_version == 'g09':
	stuff = re.search(
	    'Atomic numbers\s+I\s+N=\s+(?P<num_atoms>\d+)'
	    '\n\s+(?P<anums>.*?)'
	    'Nuclear charges.*?Current cartesian coordinates.*?\n(?P<coords>.*?)'
	    'Force Field'
	    '.*?Real atomic weights.*?\n(?P<masses>.*?)'
	    'Atom fragment info.*?Cartesian Gradient.*?\n(?P<evals>.*?)'
	    'Cartesian Force Constants.*?\n(?P<hess>.*?)'
	    'Dipole Moment',
	    open(filename, 'r').read(), flags=re.DOTALL)
    else:
	print 'Gaussian version not defined. Provide g16 or g09. Will quit'
	sys.exit()

    if stuff is not None:

	# Atomic Number, Masses, Cartesian Geometry
	anums = map(int, stuff.group('anums').split())
	anums = np.array(anums)
	masses = map(float, stuff.group('masses').split())
	masses = np.array(masses)
	coords = map(float, stuff.group('coords').split())


    if stuff is not None:

	# Atomic Number, Masses, Cartesian Geometry
	anums = map(int, stuff.group('anums').split())
	anums = np.array(anums)
	masses = map(float, stuff.group('masses').split())
	masses = np.array(masses)
	coords = map(float, stuff.group('coords').split())
	coords = [coords[i:i+3] for i in range(0, len(coords), 3)]
	coords = np.array(coords)

	natom = len(anums)	# get no of atoms by simply taking length of anums/masses array!

	evals  = np.array(map(float, stuff.group('evals').split()), dtype=float)

	# Force Constant Matrix
	low_tri = np.array(map(float, stuff.group('hess').split()), dtype=float)
	one_dim = 3 * natom
	#force_constant_matrix = np.empty([one_dim, one_dim], dtype=float)
	# NOTE: np.empty does funny stuff, and caused me grief. As always,
	#	it seems np.zeros is more reliable. Not gonna do PhD on it!
	force_constant_matrix = np.zeros([one_dim, one_dim], dtype=float)
	force_constant_matrix[np.tril_indices_from(force_constant_matrix)] = low_tri
	force_constant_matrix += np.tril(force_constant_matrix, -1).T

	return natom, anums, masses, coords, evals, force_constant_matrix

    else: raise Exception('No match found! Likely you provided a wrong fchk filename/path. Check. Try Again!')


def read_fchk_gaussian_gradient(filename):

    '''
    Read natom, cgeom, anum, amass, force-constant-matrix and also gradients!
    '''

    stuff = re.search(
	'Atomic numbers\s+I\s+N=\s+(?P<num_atoms>\d+)'
	'\n\s+(?P<anums>.*?)'
	'Nuclear charges.*?Current cartesian coordinates.*?\n(?P<coords>.*?)'
	'Force Field'
	'.*?Real atomic weights.*?\n(?P<masses>.*?)'
	'Atom fragment info.*?Cartesian Gradient.*?\n(?P<evals>.*?)'
	#'Cartesian Force Constants.*?\n(?P<hess>.*?)'
	'Dipole Moment',
	open(filename, 'r').read(), flags=re.DOTALL)

    if stuff is not None:

	# Atomic Number, Masses, Cartesian Geometry
	anums = map(int, stuff.group('anums').split())
	anums = np.array(anums)
	masses = map(float, stuff.group('masses').split())
	masses = np.array(masses)
	coords = map(float, stuff.group('coords').split())
	coords = [coords[i:i+3] for i in range(0, len(coords), 3)]
	coords = np.array(coords)

	natom = len(anums)	# get no of atoms by simply taking length of anums/masses array!

	evals  = np.array(map(float, stuff.group('evals').split()), dtype=float)

	return natom, anums, masses, coords, evals

    else: raise Exception('No match found! Likely you provided a wrong fchk filename/path. Check. Try Again!')


def read_vibronic_model_hessian(filename):

    '''
    Read Hessian from Vibron_TS_info
    '''

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
	
    # ================================
    # Read Hessian from Vibron_TS_info
    # ================================

    hessian = np.zeros((nmode, nmode))
    for line in f:
	if 'Hessian' in line:
	    hess_index = f.index(line)
	else:
	    pass
	    
    hess_cols = int(f[hess_index+1].strip().split()[2])
    hess_blocks = nmode / hess_cols

    m = 0
    start_index = hess_index + 2
    for r in range(hess_blocks):
	for i in range(nmode):
	    hessian[i,:][m:m+4] = np.array([float(k) for k in f[i+start_index].strip().split()])
	m = m + 4
	start_index = start_index + nmode + 1
    
    # Also read normal modes and frequencies from transform_cartesian_normal
    g = open('transform_cartesian_normal').readlines() 
    natom = int(g[0].strip().split()[0])
    amass = np.array([float(k) for k in g[1].strip().split()])


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

    return hessian, L_ref_mat


def read_irc_fchk_gaussian(filename):

    '''
    Read ngeom, Results(Energy), Geometries, Gradient from irc.fchk
    '''

    # Read all lines
    try:
	with open(filename) as f:
	    all_lines = f.readlines()
    except EnvironmentError:
	print 'Something not right with Gaussian IRC file.'
	print 'IOError / OSError / WindowsError. Check. Fix. Rerun.'


    # Get all indices of lines starting with "IRC"
    all_irc_index = []
    for line in all_lines:
	if line.startswith('IRC'):
	    all_irc_index.append(all_lines.index(line))
	else:
	    pass

    # Get number of "geometry variables from 4th line (and thus number of atoms)
    n_xyz = int(all_lines[all_irc_index[3]].strip().split()[-1])
    natom = n_xyz/3

    # Get Energies at all Geometries and the number of geometries in turn
    foo_temp = ''
    lines_energy = all_lines[all_irc_index[4]+1:all_irc_index[5]]
    for line in lines_energy:
	foo_temp = foo_temp + ''.join(line.strip()+' ')
    
    foo_temp = foo_temp.split()
    all_energies = np.array([float(i) for i in foo_temp][::2])
    all_energies = all_energies - np.amin(all_energies)
    
    all_grad_norm = np.array([float(i) for i in foo_temp][1::2])

    # Number of IRC points
    ngeom = len(all_energies)

    # Get All Cartesian Geometries
    foo_temp_2 = ''
    lines_geometry = all_lines[all_irc_index[5]+1:all_irc_index[6]]
    for line in lines_geometry:
	foo_temp_2 = foo_temp_2 + ''.join(line.strip()+' ')

    foo_temp_2 = foo_temp_2.split()
    geometries = np.array([float(i) for i in foo_temp_2])

    all_geometries =  geometries.reshape((ngeom, natom, 3))

    # Get All Gradient Vectors
    foo_temp_3 = ''
    lines_gradient = all_lines[all_irc_index[6]+1:all_irc_index[7]]
    for line in lines_gradient:
	foo_temp_3 = foo_temp_3 + ''.join(line.strip()+' ')

    foo_temp_3 = foo_temp_3.split()
    gradients = np.array([float(i) for i in foo_temp_3])

    all_gradients =  gradients.reshape((ngeom, natom, 3))

    return ngeom, all_geometries, all_energies, all_gradients, all_grad_norm



# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-END-GAUSSIAN-INTERFACE-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

