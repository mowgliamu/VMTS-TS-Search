# -*- coding: utf-8 -*-
#!usr/bin/env python

import re
import os
import sys
import errno
import timeit
import inspect
import logging
import platform
import datetime

import numpy as np
import cmath, math
from scipy import linalg
from pprint import pprint
from subprocess import call
from operator import itemgetter
from scipy.signal import argrelextrema
from scipy.optimize import minimize

from pprint import pprint
from copy import deepcopy
from libspec import path_exists

from alignment import *
from input_parameters import *
from gaussian_interface import *
from physical_constants import *

# Print Precision!
np.set_printoptions(precision=8, suppress=True)

#sys.tracebacklimit=0

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-CLASS-SECTION-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

class Molecule(object):

    """
    Abstraction of a molecule, primarily based on the cartesian geometry and hessian.
    Associated methods will perform the harmonic vibrational analysis and the rigid
    rotor rotational analysis. Appropriate coordinate transformations will be perfomed.
    
    =========================== ================================================
    Attribute                   Description
    =========================== ================================================
    `natom`              	Number of atoms 
    `amass`			An array of atomic masses 
    `eq_geom_cart`		Molecular geometry in cartesian coordinates
    `force_constant_matrix`	Force Constant Matrix in cartesian coordinates
    ----------------------------------------------------------------------------
    `energy`			Single point electronic energy at the given geometry
    `zpe`			The zero point energy as reported by Gaussian / ACES
    `gradient`			Gradient vector at the corresponding geometry
    `hessian`			Hessian matrix at the corresponding geometry
    `frequencies`		Harmonic Vibrational Frequencies (in Wavenumbers)
    `mwcart`			Mass-weighted cartesian goemetry
    `com`			Centre-of-Mass coordinates
    `moi`			Moment-of-Inertia Tensor
    `PointGroup`		Point Group Symmetry
    `RotationalConstants`	[A, B, C] associated with Ia, Ib, Ic
    ----------------------------------------------------------------------------
    `Lcart`			Cartesian Normal Coordinates
    `Lmwc'			Mass-weighted Cartesian Normal Coordinates
    `Ldmfs`			Dimensionless frequency scaled Normal Coordinates
    `Ltransrot`			Constructed Translation and/or Rotation Eigenvectors
    ----------------------------------------------------------------------------
    `EckartMatrix`		Eckart Rotation Matrix for a given reference geometry
    `Duschinsky`		Duschinsky rotation matrix for a given reference geometry
    `Displacement`		Displacement vector for a given reference geometry
    =========================== ================================================    
    
    ----------
    ACES NOTES:
    ----------

    Normal Coordinate Matrix from the file NORMAL_FDIF are in CARTESIAN coordinates.
    """

    fred  = 0.091135 		# The infamous fred! Conversion factor for DMNC 

    def __init__(self, natom, amass, eq_geom_cart, force_constant_matrix):

	self.natom = natom
	self.amass = amass
	self.eq_geom_cart = eq_geom_cart
	self.force_constant_matrix = force_constant_matrix

	self.mwcart   = None
	self.com      = None
	self.moi      = None
	self.energy   = None
	self.zpe      = None
	self.hessian  = None
	self.frequencies  = None
	
	self.PointGroup = None
	self.RotationalConstants = None

	self.Lcart = None
	self.Lmwnc = None
	self.Ldmfs = None
	self.Ltransrot = None

	self.EckartMatrix = None
	self.Duschinsky   = None
	self.Displacement = None
	
	# What about linear molecules?
	if project_translation and project_rotation:
	    self.nmode = 3*self.natom - 6
	elif project_translation and not project_rotation:
	    self.nmode = 3*self.natom - 3
	elif not project_translation and project_rotation:
	    self.nmode = 3*self.natom - 3
	else:
	    self.nmode = 3*self.natom

	self.gradient = np.zeros(self.nmode)

    def get_mass_weighted_cartesian_coordinates(self):

	'''
	Mass-weighted Cartesian coordinates
	'''
	
	mwcart = np.zeros((self.natom, 3))

	for i in range(self.natom):
	    mwcart[i][0] = np.sqrt(self.amass[i])*self.eq_geom_cart[i][0]
	    mwcart[i][1] = np.sqrt(self.amass[i])*self.eq_geom_cart[i][1]
	    mwcart[i][2] = np.sqrt(self.amass[i])*self.eq_geom_cart[i][2]

	self.mwcart = mwcart

	return mwcart


    def get_mw_cart_flat_array(self):

	'''
	Mass-weighted Cartesian coordinates as 3N array!
	'''

	return self.mwcart.flatten()
	

    def get_centre_of_mass_mwc(self):

	'''
	Centre of Mass using m.w. Cartesian coordinates
	'''

	TotMass    = np.sum(self.amass)
	CentreMass = np.zeros(3)

	for i in range(self.natom):
	    CentreMass[0] += np.sqrt(self.amass[i])*self.mwcart[i][0]
	    CentreMass[1] += np.sqrt(self.amass[i])*self.mwcart[i][1]
	    CentreMass[2] += np.sqrt(self.amass[i])*self.mwcart[i][2]

	CentreMass = CentreMass/TotMass
	self.com   = CentreMass

	return CentreMass


    def get_inertia_tensor(self):

	'''
	Inertia Tensor (3x3 Matrix) using m.w. Cartesian coordinates
	'''

	Inertia_Tensor = np.zeros((3,3))

	for i in range(self.natom):
	    Inertia_Tensor[0][0] += self.mwcart[i][1]*self.mwcart[i][1] + self.mwcart[i][2]*self.mwcart[i][2]
	    Inertia_Tensor[1][1] += self.mwcart[i][0]*self.mwcart[i][0] + self.mwcart[i][2]*self.mwcart[i][2]
	    Inertia_Tensor[2][2] += self.mwcart[i][0]*self.mwcart[i][0] + self.mwcart[i][1]*self.mwcart[i][1]
	    Inertia_Tensor[0][1] += self.mwcart[i][0]*self.mwcart[i][1]
	    Inertia_Tensor[0][2] += self.mwcart[i][0]*self.mwcart[i][2]
	    Inertia_Tensor[1][2] += self.mwcart[i][1]*self.mwcart[i][2]
    
	Inertia_Tensor[1][0] = Inertia_Tensor[0][1]
	Inertia_Tensor[2][0] = Inertia_Tensor[0][2]
	Inertia_Tensor[2][1] = Inertia_Tensor[1][2]

	print
	print '@get_inertia_tensor: Moment of Inertia for given geometry'
	print Inertia_Tensor
	print

	self.moi = Inertia_Tensor

	return Inertia_Tensor


    def get_eckart_frame_self(self):

	'''
	Put the molecule in its own Eckart frame! COM at Origin and X-Y-Z axes as Principal Axes.
	'''

	# Check if CM at origin, and Inertia_Tensor Diagonal. If yes: exit
	# If no: translate CM to origin, diagonalize IT, and rotate coordinates
    
	tolerance = 1e-5
	
	print
	print 'eq_geom_cart: before com frame'
	pprint(self.eq_geom_cart)
	print

	# Always put COM at origin (even if it already is, does not hurt)
	#for i in range(self.natom):
	#    self.eq_geom_cart[i:,] = self.eq_geom_cart[i:,] - self.com
	
	self.eq_geom_cart = self.eq_geom_cart - self.com.T

	print
	print 'eq_geom_cart: after com frame'
	print self.eq_geom_cart
	print

	self.com = np.zeros(3)

	# Check for MOI
	for i in range(3):
	    for j in range(3):
		if i != j:
		    if self.moi[i, j] > tolerance:
			print 'Inertia Tensor Not Digonal, Will Diagonalize'
			Ival, Ivec = np.linalg.eigh(self.moi)
			break
		    else:
			print 'Inertia Tensor already diagonal, Sort eigenvalues'
			Ival, Ivec = np.linalg.eigh(self.moi)
		else:
		    pass

	self.moi = np.diag(Ival)

	# Rotate coordinates to Principal Axes
	self.eq_geom_cart = np.dot(self.eq_geom_cart, Ivec)

	print
	print 'eq_geom_cart: principal axes'
	print self.eq_geom_cart
	print

	return Ival, Ivec


    def harmonic_vibrational_analysis(self, proj_translations=True, proj_rotations=False, verbose=False):

	'''
	Perform the normal mode analysis starting from the 3N x 3N Force Constant Matrix:

	- project out translations and/or rotations as requested
	- mass-weight force constant matrix, diagonalize (omega, L)
	- get frequencies in cm-1 (sqrt(omega)*5140.48)
	
	'''
	
	# Get the matrix of atomic masses 
	mass_matrix = np.diag(np.repeat(self.amass, 3))
	mass_matrix_sqrt_div = np.diag(np.repeat(1.0/np.sqrt(self.amass), 3))

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Cartesian Geometry as given (could be lab frame)'
	    print self.eq_geom_cart
	    print
	else:
	    pass

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: COM'
	    print self.com
	    print
	else:
	    pass

	# calculate the center of mass in cartesian coordinates
	xyzcom = self.eq_geom_cart - self.com.T

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Cartesian Geometry in COM frame'
	    print xyzcom
	    print
	else:
	    pass

	# Initialize (3N, 6) array for Translation and Rotation
	Dmat = np.zeros((3*self.natom, 6), dtype=float)

	#####################################################
	# Construct Eigenvectors correspoding to Translation#
	#####################################################

	for i in range(3):
	    for k in range(self.natom):
		for alpha in range(3):
		    if alpha == i:
			Dmat[3*k+alpha, i] = np.sqrt(self.amass[k])
		    else:
			pass

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Translation Eigenvectors'
	    print Dmat
	    print
	else:
	    pass

	###################################################
	# Construct Eigenvectors correspoding to Rotation #
	###################################################

	# 1. Get Inertia Tensor and Diagonalize
	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Moment of Inertia for given geometry'
	    print self.moi
	    print
	else:
	    pass

	Ival, Ivec = np.linalg.eigh(self.moi)
	
	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Diagonalize MOI; check Ival, Ivec'
	    print Ival
	    print Ivec
	    print
	else:
	    pass

	# 2. Construct Pmat
	Pmat = np.dot(xyzcom, Ivec)
	
	# 3. Construct Rotational Normal Coordinates
	for i in range(self.natom):
	    for j in range(3):
		Dmat[3*i + j, 3] = (Pmat[i, 1]*Ivec[j, 2] - Pmat[i, 2]*Ivec[j, 1])*np.sqrt(self.amass[i])
		Dmat[3*i + j, 4] = (Pmat[i, 2]*Ivec[j, 0] - Pmat[i, 0]*Ivec[j, 2])*np.sqrt(self.amass[i])
		Dmat[3*i + j, 5] = (Pmat[i, 0]*Ivec[j, 1] - Pmat[i, 1]*Ivec[j, 0])*np.sqrt(self.amass[i])


	##################################################################################
	# Set the orthonormalized Translation-Rotation Eigenvectors to attribute Ltransrot
	##################################################################################

	Translation = Dmat[:,0:3]
	Rotation = Dmat[:,3:6]

	# Separately orthonormalize translation and rotation
	Dtrans, xxx = np.linalg.qr(Translation)
	Drot, xxx = np.linalg.qr(Rotation)

	LTR = np.zeros((3*self.natom, 6), dtype=float)
	LTR[:,0:3] = Dtrans
	LTR[:,3:6] = Drot

	self.Ltransrot = Drot

	print
	print 'Orthonormality of Translation and Rotation Vectors'
	print np.dot(LTR.T, LTR)
	print

	# Mass-weight the force constant matrix
	mw_fcm = np.dot(mass_matrix_sqrt_div, np.dot(self.force_constant_matrix, mass_matrix_sqrt_div))
	
	# Diagonalize (Eigenvalues are sorted)
	hval, hvec = np.linalg.eigh(mw_fcm)

	print
	print '@harmonic_vibrational_analysis: All eigenvalues of FCM before  Projection'
	print np.sqrt(abs(hval))*hfreq_cm
	print

	# Project out Rotation and Translation from Hessian
	Imat = np.eye(LTR.shape[0])
	llt = np.dot(LTR, LTR.T)
	proj_trans_rot_hessian =  np.dot(Imat - llt, np.dot(mw_fcm, Imat - llt))
	rphval, rphvec = np.linalg.eigh(proj_trans_rot_hessian)

	print
	print 'Hessian eigenvalues before sorting -ve eigenvalues'
	print rphval
	print

	# SORT OUT ALL -VE FREQUENCIES
	all_index_0 = np.where(abs(rphval) < 1e-4)[0]
	eigvals_0 = rphval[all_index_0]
	eigvec_0 = rphvec[:, all_index_0]

	print
	print 'all_index_0:', all_index_0
	print 'eigvals_0:', eigvals_0
	print

	# A cleaner solution?
	rphval = np.delete(rphval, all_index_0, axis=0)
	rphvec = np.delete(rphvec, all_index_0, axis=1)
	rphval = np.concatenate([eigvals_0, rphval])
	rphvec = np.concatenate([eigvec_0, rphvec], axis=1)

	print
	print 'Hessian eigenvalues after sorting -ve eigenvalues'
	print rphval
	print

	vib_freq_cm = np.sqrt(abs(rphval[3:]))*hfreq_cm

	print
	print '@harmonic_vibrational_analysis: Eigenvalues after Translational and Rotational Projection according to Gaussian using (1-P)H(1-P)'
	print vib_freq_cm
	print

	Lmwc = np.zeros((3*self.natom, self.nmode))
	Lmwc[:,0:3] = Drot
	Lmwc[:,3:self.nmode] = rphvec[:,6:3*self.natom]

	print
	print 'Orthonormality of 3N-3 Vibration and Rotation Vectors [After Gluing]'
	print np.dot(Lmwc.T, Lmwc)
	print

	# NORMAL MODES - SET ATTRIBUTE
	self.Lmwc = Lmwc

	if add_rotational_energy:
	    rphval[3:6] = rot_freq
	    vib_freq_cm[0:3] = rot_freq_cm
	else:
	    pass

	# HESSIAN - SET ATTRIBUTE [ATOMIC UNITS AT THIS POINT]
	self.hessian = np.diagflat(rphval[3:])

	print
	print '@harmonic_vibrational_analysis: hessian - 3N-3 with 3 zero eigenvalues'
	print self.hessian
	print

	# FREQUENCIES - SET ATTRIBUTE
	self.frequencies = vib_freq_cm

	return vib_freq_cm, Lmwc


    def harmonic_vibrational_analysis_2(self, proj_translations=True, proj_rotations=False, verbose=False):

	'''
	Perform the normal mode analysis starting from the 3N x 3N Force Constant Matrix:

	- project out translations and/or rotations as requested
	- mass-weight force constant matrix, diagonalize (omega, L)
	- explicitly project out translation and get 3N-3 normal modes using lowdin orthogonalization
	- get frequencies in cm-1 (sqrt(omega)*5140.48)
	- get dimensionless normal coordinates (using fred!)
	
	'''
	
	# Get the matrix of atomic masses 
	mass_matrix = np.diag(np.repeat(self.amass, 3))
	mass_matrix_sqrt_div = np.diag(np.repeat(1.0/np.sqrt(self.amass), 3))

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Cartesian Geometry as given (could be lab frame)'
	    print self.eq_geom_cart
	    print
	else:
	    pass

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: COM'
	    print self.com
	    print
	else:
	    pass

	# calculate the center of mass in cartesian coordinates
	xyzcom = self.eq_geom_cart - self.com.T

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Cartesian Geometry in COM frame'
	    print xyzcom
	    print
	else:
	    pass

	# Initialize (3N, 6) array for Translation and Rotation
	Dmat = np.zeros((3*self.natom, 6), dtype=float)

	#####################################################
	# Construct Eigenvectors correspoding to Translation#
	#####################################################

	for i in range(3):
	    for k in range(self.natom):
		for alpha in range(3):
		    if alpha == i:
			Dmat[3*k+alpha, i] = np.sqrt(self.amass[k])
		    else:
			pass

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Translation Eigenvectors'
	    print Dmat
	    print
	else:
	    pass

	###################################################
	# Construct Eigenvectors correspoding to Rotation #
	###################################################

	# 1. Get Inertia Tensor and Diagonalize
	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Moment of Inertia for given geometry'
	    print self.moi
	    print
	else:
	    pass

	Ival, Ivec = np.linalg.eigh(self.moi)
	
	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Diagonalize MOI; check Ival, Ivec'
	    print Ival
	    print Ivec
	    print
	else:
	    pass

	# 2. Construct Pmat
	Pmat = np.dot(xyzcom, Ivec)
	
	# 3. Construct Rotational Normal Coordinates
	for i in range(self.natom):
	    for j in range(3):
		Dmat[3*i + j, 3] = (Pmat[i, 1]*Ivec[j, 2] - Pmat[i, 2]*Ivec[j, 1])*np.sqrt(self.amass[i])
		Dmat[3*i + j, 4] = (Pmat[i, 2]*Ivec[j, 0] - Pmat[i, 0]*Ivec[j, 2])*np.sqrt(self.amass[i])
		Dmat[3*i + j, 5] = (Pmat[i, 0]*Ivec[j, 1] - Pmat[i, 1]*Ivec[j, 0])*np.sqrt(self.amass[i])


	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Translation and Rotation Eigenvectors'
	    print Dmat
	    print
	else:
	    pass


	##################################################################################
	# Set the orthonormalized Translation-Rotation Eigenvectors to attribute Ltransrot
	##################################################################################


	LTR, xxx = np.linalg.qr(Dmat)	# Numpy's QR decompsition. Gram-Schmidt, basically.
	#self.Ltransrot = LTR
	self.Ltransrot = Dmat

	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Orthonormalized Translation and Rotation Eigenvectors'
	    print LTR
	    print
	else:
	    pass

	# ==================================================
	# Orthogonalization (QR Decomposition: Gram-Schmidt)
	# ==================================================

	if proj_translations and proj_rotations:
	    q, _ = np.linalg.qr(Dmat)
	elif proj_translations and not proj_rotations:
	    q, _ = np.linalg.qr(Dmat[:, :3])
	elif not proj_translations and proj_rotations:
	    q, _ = np.linalg.qr(Dmat[:, 3:])
	else:
	    print '@harmonic_vibrational_analysis:'
	    print 'Separation of Translation and Rotation not requested'
	    print 'Make sure if this is what you want. Will return Full Hessian'
	    q = np.zeros((3*self.natom, 6), dtype=float)
	
	if verbose:
	    print
	    print '@harmonic_vibrational_analysis: Translation-Rotation Eigenvectors after QR Decomposition'
	    print q
	    print
	    print 'Test orthogonality of this Dmat'
	    print np.dot(q.T, q)
	    print
	else:
	    pass

	# =====================================================================================
	# Project out requested degrees of freedom from the mass-weighted force constant matrix
	# =====================================================================================

	# Identity Matrix
	Imat = np.eye(q.shape[0])

	# Projection Operator: Q * Q.T
	qqp = np.dot(q, q.T)
	#qqp = np.dot(LTR, LTR.T)
	
	# Mass-weight the force constant matrix
	mw_fcm = np.dot(mass_matrix_sqrt_div, np.dot(self.force_constant_matrix, mass_matrix_sqrt_div))
	
	# Diagonalize (Eigenvalues are sorted)
	hval, hvec = np.linalg.eigh(mw_fcm)

	# Project the force constant matrix (1 - P)*H*(1 - P)
	proj_hessian =  np.dot(Imat - qqp, np.dot(mw_fcm, Imat - qqp))
	
	# Diagonalize (Eigenvalues are sorted)
	phval, phvec = np.linalg.eigh(proj_hessian)
	
	print 
	print '@harmonic_vibrational_analysis: hval (before projection)'
	print hval
	print
	
	print
	print '@harmonic_vibrational_analysis: All eigenvalues of FCM before Projection'
	print np.sqrt(abs(hval))*hfreq_cm
	print

	#=====================
	## NEW: LOWDIN TEST ##
	#=====================

	# 1. (1 - P_translation)*hvec
	U_dagger = np.dot((Imat - qqp), phvec)

	# 2. Compute S matrix
	S_mat = np.dot(U_dagger.T, U_dagger)

	print
	print 'The overlap S matrix'
	print S_mat
	print

	# 3. Diagona;ize S matrix
	sval, svec  = np.linalg.eigh(S_mat)

	print
	print 'Eigenvalues of S matrix: 3 zero eigenvalues?'
	print sval
	print
	print 'Check orthogonality of eigenvectord of S matrix'
	print np.dot(svec.T, svec)
	print

	shalf = np.zeros(3*self.natom)
	shalf[3:] = sval[3:]**(-0.5)

	# 4. Compute S^(-1/2)
	S_mat_half = np.dot(svec, np.dot(np.diag(shalf), svec.T))

	# 5. Calculate V = U_dagger * S_half
	V_mat = np.dot(U_dagger, S_mat_half)

	print
	print 'V_mat: U_dagger * S_half_mat [Transformed Eigenvectors (Lowdin basis)]'
	pprint(V_mat.T)
	print

	# Search (and remove) for column vectors of V, which are rigorously zero
	index_translation = []
	for i in range(3*self.natom):
	    if np.all(abs(V_mat[:,i]) < 1e-6):
		index_translation.append(i)
	    else:
		pass

	# These will be the new orthonormal normal modes!
	V_mat = np.delete(V_mat, index_translation, axis=1)

	# NORMAL MODES - SET ATTRIBUTE
	self.Lmwc = V_mat

	print
	print 'V_mat: Check Orthogonality'
	pprint(np.dot(V_mat.T, V_mat))
	print

	# Transform Hessian (MW) in V basis - MUST BE DIAGONAL
	hess_V = np.dot(V_mat.T, np.dot(mw_fcm, V_mat))

	# HESSIAN - SET ATTRIBUTE
	self.hessian = hess_V

	# Process frequencies
	hval = np.diag(hess_V)
	vib_freq_cm = np.sqrt(abs(hval))*hfreq_cm

	print 
	print '@harmonic_vibrational_analysis: hval (after projection)'
	print hval
	print
	
	print
	print '@get_normal_coordinates: All eigenvalues of FCM after Projection'
	print vib_freq_cm
	print

	# FREQUENCIES - SET ATTRIBUTE
	self.frequencies = vib_freq_cm

	return vib_freq_cm, V_mat





    def X_to_Q(self, inp_cart, ref_cart):

       	'''
	Transformation between Cartesian and Normal Coordinates
	'''

	mwcart = np.zeros((self.natom, 3))

	for i in range(self.natom):
	    mwcart[i][0] = np.sqrt(self.amass[i])*(inp_cart[i][0] - ref_cart[i][0])
	    mwcart[i][1] = np.sqrt(self.amass[i])*(inp_cart[i][1] - ref_cart[i][1])
	    mwcart[i][2] = np.sqrt(self.amass[i])*(inp_cart[i][2] - ref_cart[i][2])

	mwnc_from_mwcart = np.dot(self.Lmwc.T, mwcart.flatten())

	return mwnc_from_mwcart


    def Q_to_X(self, inp_mwnc, ref_cart):

	'''
	Transformation between Cartesian and Normal Coordinates
	'''

	# Get the matrix of atomic masses 
	mass_matrix = np.diag(np.repeat(self.amass, 3))
	mass_matrix_sqrt_div = np.diag(np.repeat(1.0/np.sqrt(self.amass), 3))
       
	# Transform
	cart_from_mwnc = np.dot(mass_matrix_sqrt_div, np.dot(self.Lmwc, inp_mwnc)) + ref_cart.flatten()

	# Reshape
	cart_from_mwnc = np.reshape(cart_from_mwnc, (self.natom, 3))

	return cart_from_mwnc	


    def get_dimensionless_normal_coordinates(self, inp_cart, ref_cart):

	'''
	Use fred to go from MWNC to dimensionless normal coordinates
	'''

	# The Transformation Matrix : D~ [MCTDH Notation] (3N - 3, 3N)
	xdmfs = np.zeros(np.shape(self.Lmwc)).T
	
	for alpha in range(self.nmode):
	    for i in range(self.natom):	
		for j in range(3):
		    k = 3*i + j
		    xdmfs[alpha][k] = (fred*np.sqrt(self.frequencies[alpha])*np.sqrt(self.amass[i]))*self.Lmwc.T[alpha][k]

	# X - X0
	relative_cartesian = inp_cart.flatten() - ref_cart.flatten()
	
	# Q = D~ * (X - X0)
	Q = np.dot(xdmfs, relative_cartesian)
    
	return Q, xdmfs


    def get_geometry_cartesian_coordinates(self, inp_dmfs, ref_cart):

	'''
	Get cartesian geometry for a given dimensionless coordinate
	'''

	# The Transformation Matrix: D~' [MCTDH Notation] (3N, 3N - 3)
	xdmfs = np.zeros(np.shape(self.Lmwc))
	
	for alpha in range(self.nmode):
	    for i in range(self.natom):	
		for j in range(3):
		    k = 3*i + j
		    xdmfs[k][alpha] = self.Lmwc[k][alpha]/(fred*np.sqrt(self.hessian[alpha][alpha])*np.sqrt(self.amass[i]))

	# X = X0 + D~'*Q
	X = ref_cart.flatten() + np.dot(xdmfs, inp_dmfs)

	return X, xdmfs



    def zero_order_axis_switch(self, other):

	'''
	Dymarsky and Kudin's generalized solution to compute pseudorotation matrix.
	J. Chem. Phys. 122, 124103, 2005; Based on Pickett & Strauss, JACS, 1970.

	Decided not to implement Anirban's approach, but Ozkan's paper is worth a read:
	Ilker Ozkan, Journal of Molecular Spectroscopy, 139, 147, 1990

	Also this method is a generalization of Anirban's scheme implemented in VIBRON:
	J. Phys. Chem A, 121, 5326 (Based on Ozkan and Hougen & Watson)

	self:	target state
	other:	reference state
	'''
	
	C_matrix = np.zeros((3,3))

	for i in range(self.natom):
	    C_matrix[0][0] += other.mwcart[i][0]*self.mwcart[i][0]	
	    C_matrix[0][1] += other.mwcart[i][0]*self.mwcart[i][1]	
	    C_matrix[0][2] += other.mwcart[i][0]*self.mwcart[i][2]	
	    C_matrix[1][0] += other.mwcart[i][1]*self.mwcart[i][0]	
	    C_matrix[1][1] += other.mwcart[i][1]*self.mwcart[i][1]	
	    C_matrix[1][2] += other.mwcart[i][1]*self.mwcart[i][2]	
	    C_matrix[2][0] += other.mwcart[i][2]*self.mwcart[i][0]	
	    C_matrix[2][1] += other.mwcart[i][2]*self.mwcart[i][1]	
	    C_matrix[2][2] += other.mwcart[i][2]*self.mwcart[i][2]	

	print
	print 'A matrix of Dymarsky and Kudin'
	print C_matrix
	print

	# Calculate C' * C (A1) and C * C' (A2)
	C_T_C = np.dot(C_matrix.T, C_matrix)
	C_C_T = np.dot(C_matrix, C_matrix.T)

	print
	print 'The A1 and A2 matrix of Dymarksy and Kudin'
	print C_T_C
	print
	print C_C_T
	print

	# Diagonalize A1 and A2
	a1_val, a1_vec = np.linalg.eigh(C_T_C)
	a2_val, a2_vec = np.linalg.eigh(C_C_T)

	# Calculate T matrix
	T_mat = np.dot(a2_vec, a1_vec)

	print
	print 'The T matrix of Dymarksy and Kudin'
	print T_mat
	print

	# Check that C * T is a symmetric matrix (this proves Eckart conditions are satisfied)
	S_mat = np.dot(C_matrix, T_mat)
	print
	print 'The S matrix of Dymarksy and Kudin'
	print S_mat
	print
		    
	print
	print 'Determinant of T rotation matrix, should be +1.0'
	print np.linalg.det(T_mat)
	print

	return T_mat


    def get_duschinsky_displacement_after_eckart(self, other, eckart_b_matrix):

	'''
	"Modified" Duschinsky matrix and Displacement vector!

	J = L_initial.T * (B_mat) * L_final
	K = L_initial.T * M^(-1/2) * ECKART_ROTATION * (cart_final - cart_initial)
	'''

	# Get the matrix of atomic masses 
	mass_matrix = np.diag(np.repeat(self.amass, 3))
	mass_matrix_sqrt = np.diag(np.repeat(np.sqrt(self.amass), 3))
	
	# Cartesian Displacement
	displacement = self.eq_geom_cart.flatten() - other.eq_geom_cart.flatten()

	# Duschinsky matrix
	J_dus = np.dot(self.Lmwc.T, np.dot(eckart_b_matrix.T, other.Lmwc))
	
	# Displacement vector
	right_product = np.dot(eckart_b_matrix, self.eq_geom_cart.flatten()) - other.eq_geom_cart.flatten()
	left_product  = np.dot(other.Lmwc.T, mass_matrix_sqrt)
	K_dis = np.dot(left_product, right_product)

	return J_dus, K_dis

    
    def get_transformed_abinitio_data(self, other, dus_mat, dis_vec):

	'''
	q_Ar   = J_A * q_A + K_A
	E_Ar   = E_A - g_Ar.T 
	g_Ar.T = g_A.T * {J_A}^-1 
	H_Ar   = {J_A.T}^-1 * H_A * {J_A}^-1
	'''

	# ===========================
	# Coordinate Transformations!
	# ===========================

	E_Ar  = self.energy
	g_Ar  = np.dot(dus_mat, self.gradient)	
	H_Ar  = np.dot(dus_mat.T, np.dot(self.hessian, dus_mat))

	print
	print 'Diagonalize the transformed Hessian and check eigenvalues'
	hval, hvec = np.linalg.eigh(H_Ar)
	print 'new frequencies'
	print np.sqrt(np.abs(hval/au_to_ev))*hfreq_cm
	print
	print 'new gradients'
	print g_Ar
	print

	if dimensionless:
	    
	    # Dimensionless coordinate of minima in its own system (always zero!)
	    Q_A, D_A  = self.get_dimensionless_normal_coordinates(self.eq_geom_cart, self.eq_geom_cart)
	    
	    # Displacement vector in reduced dimensionless coordinates (from MWNC) [Scaled as fred*sqrt(omega)]
	    dis_Q = np.zeros(nmode)
	    for i in range(nmode):
		dis_Q[i] = fred*(np.sqrt(other.frequencies[i]))*dis_vec[i]

	    # New minima position in reference state coordinate system (reduced dimensionless)
	    q_Ar  = np.dot(dus_mat, Q_A) + dis_Q

	    # Transform Gradient to dimensionless coordinates
	    gradient_dimless = np.zeros(nmode)
	    for i in range(nmode):
		gradient_dimless[i] = (1.0/(fred*(np.sqrt(other.frequencies[i]))))*(g_Ar[i])

	    # Transform Hessian to dimensionless coordinates
	    hessian_dimless = np.zeros((nmode, nmode))
	    for i in range(nmode):
		for j in range(nmode):
		    hessian_dimless[i][j] = (1.0/(fred*(np.sqrt(other.frequencies[i]))))*(H_Ar[i][j])*(1.0/(fred*(np.sqrt(other.frequencies[j]))))

	    return q_Ar, E_Ar, gradient_dimless, hessian_dimless
	    
	else:

	    # Normal coordinate self (always zero)
	    Q_A   = self.X_to_Q(self.eq_geom_cart, self.eq_geom_cart)

	    # New minima position (normal coordinate)
	    q_Ar  = np.dot(dus_mat, Q_A) + dis_vec
	
	    return q_Ar, E_Ar, g_Ar, H_Ar





class Potential(object):

    """
    A multi-dimensional Quadratic Potential which is a smooth
    function of nulcear coordinates (Taylor series expansion).
    
    V(q) = E' + g'.T * (q - q') + 1/2 * (q - q').T * H' (q - q')

    =========================== ===============================================================
    Attribute                   Description
    =========================== ===============================================================
    `position`                  Reference geometry in dimensionless normal coordinates (vector)
    `energy`                    Electronic energy at the reference geometry 	       (scalar)
    `gradient`             	Gradient vector at the reference geometry 	       (vector)
    `hessian`         		Hessian matrix at the reference geometry	       (matrix)
    =========================== ===============================================================    
    
    """

    def __init__(self, position, energy, gradient, hessian):

        self.position =  position      # vector
        self.energy   =  energy        # scalar
        self.gradient =  gradient      # vector
        self.hessian  =  hessian       # matrix

    def getPosition(self):
        ''' Returns coordinates of minima / transition state'''

        return self.position

    def getEnergy(self):
        ''' Returns vertical excitation energy''' 

        return self.energy

    def getGradient(self):
        ''' Returns gradient of energy wrt q'''

        return self.gradient

    def getHessian(self):
        ''' Returns hessian of energy wrt q'''

        return self.hessian


    def getPotential(self, q):

        ''' Returns array of potential value on the given grid'''

        energy = self.energy
        position = self.position
        gradient = self.gradient
        hessian = self.hessian

        pot_function = energy + np.dot((q - position), gradient.transpose()) +\
                                0.5*np.dot((q - position).transpose(), np.dot(hessian, (q - position)))

        return pot_function


    def getWindow(self, q, window_parameters):

        ''' Returns array of window function on the given grid'''

	# Unzip window parameters
	width = window_parameters[0]
	beta_1  = window_parameters[1]
	beta_2  = window_parameters[2]
	b_vector_1 = window_parameters[3]
	b_vector_2 = window_parameters[4]
	delta = window_parameters[5]

        position = self.position

	# Width Matrix
	if dimensionless:
	    width_matrix = np.diagflat([1.0]*nmode)						# Dimensionless Coordinates!
	else:
	    if not scale_window:
		width_matrix = np.diagflat([1.0]*nmode)						# Dimensionless Coordinates!
	    else:
		width_matrix = np.zeros((nmode, nmode))
		for i in range(nmode):
			width_matrix[i][i] = (fred*np.sqrt(reference_frequencies[i]))**2	# Mass-weighted Coordinates!


	# beta*b.T*(Q - Qts)
	linear_term = np.dot(beta_1*b_vector_1, (q-position)) +\
		      np.dot(beta_2*b_vector_2, (q-position))

	# (Q - Qts).T * A_MAT * (Q - Qts)
	quadratic_product = np.dot((q-position).transpose(), np.dot(width_matrix, (q-position)))

	# delta * [(Q - Qts).T * (Q - Qts)]**2.0
	quartic_term = (np.dot((q-position).transpose(), (q-position)))**2.0

	# exp(linear - quadratic)
        window_function = np.exp(linear_term - 0.5*width*quadratic_product - delta*quartic_term)

        return window_function


    def get_gradient_general(self, q):

	''' Returns gradient vector at a general grid point '''
	
	displacement_vec = q - self.getPosition()
	gradient_general = self.getGradient() + np.dot(self.getHessian(), displacement_vec)

	return gradient_general
    

    def get_gradient_window(self, q, window_parameters):

	''' Returns gradient vector at a general grid point with a Gaussian Window '''
	
	# Unzip window parameters
	width = window_parameters[0]
	beta_1  = window_parameters[1]
	beta_2  = window_parameters[2]
	b_vector_1 = window_parameters[3]
	b_vector_2 = window_parameters[4]
	delta = window_parameters[5]

	width_matrix = get_width_matrix(window_parameters)
	displacement_vec = q - self.getPosition()
	distance = np.dot(displacement_vec.T, displacement_vec)

	gradient_general = self.getGradient() + np.dot(self.getHessian(), displacement_vec)
	gradient_window_term = (beta_1*b_vector_1 + beta_2*b_vector_2 -\
				np.dot(width_matrix, displacement_vec) -\
				4.0*delta*distance*displacement_vec)*self.getPotential(q)

	gradient_window = (gradient_general + gradient_window_term)*self.getWindow(q, window_parameters)	

	return gradient_window


    def get_hessian_window(self, q, window_parameters):

	''' Returns hessian  at a general grid point with a Gaussian Window '''
	
	# Unzip window parameters
	width = window_parameters[0]
	beta_1  = window_parameters[1]
	beta_2  = window_parameters[2]
	b_vector_1 = window_parameters[3]
	b_vector_2 = window_parameters[4]
	delta = window_parameters[5]

	# Initialize
	hessian_window = np.zeros((nmode, nmode))

	# window value and displacement vector
	width_matrix = get_width_matrix(window_parameters)
	window_q = self.getWindow(q, window_parameters)
	displacement_vec = q - self.getPosition()
	distance = np.dot(displacement_vec.T, displacement_vec)

	# potential, gradient, hessian (general, no window)	
	hessian_self = self.getHessian()
	q_potential  = self.getPotential(q)
	grad_general = self.get_gradient_general(q)
	gradient_window_term = (beta_1*b_vector_1 + beta_2*b_vector_2 -\
				np.dot(width_matrix, displacement_vec) -\
				4.0*delta*distance*displacement_vec)*q_potential
	grad_window = (grad_general + gradient_window_term)*window_q

	
	# Build hessian
        for i in range(nmode):
	    # The "Z" terms [Window first derivative terms]
	    Z_i = beta_1*b_vector_1[i] + beta_2*b_vector_2[i] -\
		  width_matrix[i][i]*displacement_vec[i] -\
		  4.0*delta*distance*displacement_vec[i]
	    for j in range(nmode):
		# The "Z" terms [Window first derivative terms]
		Z_j = beta_1*b_vector_1[j] + beta_2*b_vector_2[j] -\
		      width_matrix[j][j]*displacement_vec[j] -\
		      4.0*delta*distance*displacement_vec[j]

		# Window second derivative terms
		if i == j:
		    window_second_derivative = - width_matrix[i][i] - 4.0*delta*(distance + 2.0*(displacement_vec[i]**2.0))
		else:
		    window_second_derivative = - width_matrix[i][j] - 8.0*delta*displacement_vec[i]*displacement_vec[j]

		# Final expression
		hessian_window[i][j] = (hessian_self[i][j] + grad_general[i]*Z_j + grad_general[j]*Z_i\
					+ (Z_i*Z_j + window_second_derivative)*q_potential)*window_q


	return hessian_window


# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-END-CLASS-SECTION-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# ----------------------- DRIVERS ---------------------
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-



def driver_process_abinitio_data(quantum_chemistry, data_minima, data_tstate):

    '''
    Driver function to process raw abinitio data obtained from GAUSSIAN / ORCA.
    ''' 

    # Process All Minima and TS
    object_minima = deepcopy(data_minima)
    object_tstate = deepcopy(data_tstate)

    # Loop over all minima and ts using a, b, nel like you do in create_diabats!!!
    for a in range(nel):
	for b in range(nel):
	    if a == b:
		print
		print 'Processing ab initio data from file for minima', data_minima[a][0]
		print

		if quantum_chemistry == 'ACES':
		    natom, amass, cgeom = read_eq_geom_aces(data_minima[a][1])
		    fcm = read_fcmfinal_aces(data_minima[a][2])

		    # create instance of Molecule
		    object_minima[a] = [Molecule(natom, amass, cgeom, fcm)]
		    current_specie = object_minima[a][0]

		    # set attribute for energy and zpe
		    setattr(current_specie, 'energy', data_minima[a][3])
		    setattr(current_specie, 'zpe', data_minima[a][4])

		elif quantum_chemistry == 'GAUSSIAN':
		    E0, ZPE = read_log_gaussian(data_minima[a][1])
		    natom, anums, amass, cgeom, gradient, fcm = read_fchk_gaussian(data_minima[a][2], gaussian_version)

		    # create instance of Molecule
		    object_minima[a] = [Molecule(natom, amass, cgeom, fcm)]
		    current_specie = object_minima[a][0]

		    # Get the matrix of atomic masses 
		    mass_matrix = np.diag(np.repeat(amass, 3))
		    mass_matrix_sqrt_div = np.diag(np.repeat(1.0/np.sqrt(amass), 3))

		    # Mass-weigthed carteisan gradient (amu^-0.5*eV)
		    gradient_mw = np.dot(mass_matrix_sqrt_div, gradient*au_to_ev)

		    # set attribute for energy and zpe
		    setattr(current_specie, 'energy', E0)
		    setattr(current_specie, 'zpe', ZPE)

		else: raise Exception('Provide a proper keyword for EST program. Error somewhere around')

		# mass-weighted cartesian coordinates
		current_specie.get_mass_weighted_cartesian_coordinates()

		# centre of mass
		current_specie.get_centre_of_mass_mwc()

		# moment of inertia tensor
		current_specie.get_inertia_tensor()

		# normal coordinates [HARMONIC VIBRATIONAL ANALYSIS]
		#old_vib_freq_cm, current_specie_lmwc = current_specie.harmonic_vibrational_analysis(project_translation, project_rotation, True)
		old_vib_freq_cm, current_specie_lmwc = current_specie.harmonic_vibrational_analysis_2(project_translation, project_rotation, True)

		# Test if normal modes are orthonormal
		if np.allclose(np.dot(current_specie_lmwc.T, current_specie_lmwc), np.identity(current_specie.nmode)):
		    pass
		else:
		    raise ValueError('Minima normal modes not orthonormal. Exiting...')


		# project out rotations and translations from mass-weighted gradient
		LTR = current_specie.Ltransrot
		Imat = np.eye(LTR.shape[0])
		llt = np.dot(LTR, LTR.T)

		print
		print 'Mass-weighted gradient before projecting out rotations'
		print gradient_mw
		print

		gradient_mw = np.dot(gradient_mw.T, (Imat - llt))

		print
		print 'Mass-weighted gradient after projecting out rotations'
		print gradient_mw
		print

		# transform gradient to normal mode coordinates
		gradient_nm = np.dot(current_specie_lmwc.T, gradient_mw)
		setattr(current_specie, 'gradient', gradient_nm)
		#setattr(current_specie, 'gradient', gradient_mw)

		print
		print 'Normal mode gradient after projecting out rotations'
		print gradient_nm
		print

		# convert hessian to new units [eV / amu * Bohr^2]
		hessian_new = getattr(current_specie, 'hessian')*hess_fact		
		setattr(current_specie, 'hessian', hessian_new)

		print
		print '@driver_abinitio: frequncies (cm-1)'
		print old_vib_freq_cm
		print

	    elif a != b and a > b:
		k = (nel*(nel)/2) - (nel-a)*((nel-a))/2 + b - a - 1	#  TS Locate
		# TODO MULTIPLE TS IMPLEMENTATION
		print
		print 'Processing ab initio data from file for TS', data_tstate[k][0]
		print

		if quantum_chemistry == 'ACES':
		    natom, amass, cgeom = read_eq_geom_aces(data_tstate[k][1])
		    fcm = read_fcmfinal_aces(data_tstate[k][2])

		    # create instance of Molecule
		    object_tstate[k] = [Molecule(natom, amass, cgeom, fcm)]
		    current_specie_ts = object_tstate[k][0]

		    # set attribute for energy and zpe
		    setattr(current_specie_ts, 'energy', data_tstate[k][3])
		    setattr(current_specie_ts, 'zpe', data_tstate[k][4])

		elif quantum_chemistry == 'GAUSSIAN':
		    E0, ZPE = read_log_gaussian(data_tstate[k][1])
		    natom, anums, amass, cgeom, gradient, fcm = read_fchk_gaussian(data_tstate[k][2], gaussian_version)

		    # create instance of Molecule [Multiple TS Implementation required]
		    object_tstate[k] = [Molecule(natom, amass, cgeom, fcm)]
		    current_specie_ts = object_tstate[k][0]

		    # Get the matrix of atomic masses 
		    mass_matrix = np.diag(np.repeat(amass, 3))
		    mass_matrix_sqrt_div = np.diag(np.repeat(1.0/np.sqrt(amass), 3))

		    # Mass-weigthed cartesian gradient
		    gradient_mw = np.dot(mass_matrix_sqrt_div, gradient*au_to_ev)

		    # set attribute for energy and zpe
		    setattr(current_specie_ts, 'energy', E0)
		    setattr(current_specie_ts, 'zpe', ZPE)

		else: raise Exception('Provide a proper keyword for EST program. Error somewhere around')

		# mass-weighted cartesian coordinates
		current_specie_ts.get_mass_weighted_cartesian_coordinates()

		# centre of mass
		current_specie_ts.get_centre_of_mass_mwc()

		# moment of inertia tensor
		current_specie_ts.get_inertia_tensor()

		# normal coordinates [HARMONIC VIBRATIONAL ANALYSIS]
		#old_vib_freq_cm, current_specie_lmwc = current_specie_ts.harmonic_vibrational_analysis(project_translation, project_rotation, True)
		old_vib_freq_cm, current_specie_lmwc = current_specie_ts.harmonic_vibrational_analysis_2(project_translation, project_rotation, True)

		# Test if normal modes are orthonormal
		if np.allclose(np.dot(current_specie_lmwc.T, current_specie_lmwc), np.identity(current_specie_ts.nmode)):
		    pass
		else:
		    raise ValueError('TS normal modes not orthonormal. Exiting...')


		# project out rotations and translations from mass-weighted gradient
		LTR = current_specie_ts.Ltransrot
		Imat = np.eye(LTR.shape[0])
		llt = np.dot(LTR, LTR.T)

		print
		print 'Mass-weighted gradient before projecting out rotations'
		print gradient_mw
		print

		gradient_mw = np.dot(gradient_mw.T, (Imat - llt))

		print
		print 'Mass-weighted gradient after projecting out rotations'
		print gradient_mw
		print

		# transform gradient to normal mode coordinates
		gradient_nm = np.dot(current_specie_lmwc.T, gradient_mw)
		setattr(current_specie_ts, 'gradient', gradient_nm)

		# convert hessian to new units [eV / amu * Bohr^2]
		hessian_new = getattr(current_specie_ts, 'hessian')*hess_fact		
		setattr(current_specie_ts, 'hessian', hessian_new)

		print
		print '@driver_abinitio: frequncies (cm-1)'
		print old_vib_freq_cm
		print
	    else:
		pass

    # ======================================================================
    # Change Electronic Energies from Absolute to Relative and convert to eV
    # ======================================================================

    all_E = []
    for a in range(nel):
	for b in range(nel):
	    if a == b:
		all_E.append(getattr(object_minima[a][0], 'energy'))
	    elif a != b and a > b:
		k = (nel*(nel)/2) - (nel-a)*((nel-a))/2 + b - a - 1	#  TS Locate
		all_E.append(getattr(object_tstate[k][0], 'energy'))
	    else:
		pass
    
    min_E = np.min(all_E)
    for a in range(nel):
	for b in range(nel):
	    if a == b:
		current_energy = getattr(object_minima[a][0], 'energy')
		setattr(object_minima[a][0], 'energy', (current_energy - min_E)*au_to_ev)
	    elif a != b and a > b:
		k = (nel*(nel)/2) - (nel-a)*((nel-a))/2 + b - a - 1	#  TS Locate
		current_energy = getattr(object_tstate[k][0], 'energy')
		setattr(object_tstate[k][0], 'energy', (current_energy - min_E)*au_to_ev)
	    else:
		pass


    return object_minima, object_tstate



def driver_process_coordinate_transformation(dic_minima, dic_tstate, reference_state):

    '''
    Eckart stuff, return ab initio data in a specific coordinate system (reference)

    dic_minima: All minima instances of type Molecule
    dic_tstate: All tstate instances of type Molecule
    reference_state: Reference specie, express other specie in its normal coordinates
    '''

    global natom
    global amass
    global nmode

    natom = dic_minima[0][0].natom
    amass = dic_minima[0][0].amass
    nmode = dic_minima[0][0].nmode

    # Create new dictionaries for Potential objects
    ab_initio_minima = deepcopy(dic_minima)
    ab_initio_tstate = deepcopy(dic_tstate)
    
    # Separate out reference state first
    if reference_state[0] == 'TS':
	ref_specie = dic_tstate[reference_state[1]][reference_state[2]]
    elif reference_state[0] == 'MIN':
	ref_specie = dic_minima[reference_state[1]][0]
    else:
	pass
    
    ref_cart = getattr(ref_specie, 'eq_geom_cart')

    # Loop over all species, process transformation
    for a in range(nel):
	for b in range(nel):
	    if a == b:

		# Coordinate Transformation
		target_cart = getattr(dic_minima[a][0], 'eq_geom_cart')
		eckart_rotation, b_matrix = eckart_rotation_general(natom, amass, target_cart, ref_cart)
		duschinsky, displacement  = dic_minima[a][0].get_duschinsky_displacement_after_eckart(ref_specie, b_matrix)
		qnew, Enew, gnew, Hnew    = dic_minima[a][0].get_transformed_abinitio_data(ref_specie, duschinsky, displacement)

		print
		print '@driver_process_coordinate_transformation: eckart rotation matrix for minima', a
		print eckart_rotation
		print
		eck_val, eck_vec = np.linalg.eigh(eckart_rotation)
		print 'Eigenvalues of Eckart rotation matrix'
		print eck_val
		print
		print '@driver_process_coordinate_transformation: duschinsky matrix for minima', a
		print duschinsky
		print
		print '@driver_process_coordinate_transformation: determinant of duschinsky matrix for minima', a
		print np.linalg.det(duschinsky)
		print
		print '@driver_process_coordinate_transformation: original hessian for minima', a
		print dic_minima[a][0].hessian
		print
		print '@driver_process_coordinate_transformation: transformed hessian for minima', a
		print Hnew
		print

		# Update attributes
		setattr(dic_minima[a][0], 'EckartMatrix', eckart_rotation)
		setattr(dic_minima[a][0], 'Duschinsky', duschinsky)
		setattr(dic_minima[a][0], 'Displacement', displacement)

		# Create new dictionary object
		current_min_pot_object = Potential(qnew, Enew, gnew, Hnew)
		ab_initio_minima[a][0] = deepcopy(current_min_pot_object)

	    elif a != b and a > b:
		k = (nel*(nel)/2) - (nel-a)*((nel-a))/2 + b - a - 1	# TS Locate
		
		# Coordinate Transformation
		target_cart = getattr(dic_tstate[k][0], 'eq_geom_cart')
		eckart_rotation, b_matrix = eckart_rotation_general(natom, amass, target_cart, ref_cart)
		duschinsky, displacement  = dic_tstate[k][0].get_duschinsky_displacement_after_eckart(ref_specie, b_matrix)
		qnew, Enew, gnew, Hnew    = dic_tstate[k][0].get_transformed_abinitio_data(ref_specie, duschinsky, displacement)

		# Update attributes
		setattr(dic_tstate[k][0], 'EckartMatrix', eckart_rotation)
		setattr(dic_tstate[k][0], 'Duschinsky', duschinsky)
		setattr(dic_tstate[k][0], 'Displacement', displacement)

		# Create new dictionary object
		current_ts_pot_object    = Potential(qnew, Enew, gnew, Hnew)
		ab_initio_tstate[k][0] = deepcopy(current_ts_pot_object)
		

    return ab_initio_minima, ab_initio_tstate


# ================
# HELPER FUNCTIONS
# ================

def get_width_matrix(window_parameters):

    '''
    '''

    width = window_parameters[0]

    # Window Coordinates Scaling
    if dimensionless:
	width_matrix = np.diagflat([1.0*width]*nmode)
    else:
	if not scale_window:
	    width_matrix = np.diagflat([1.0*width]*nmode)	
	else:
	    width_matrix = np.zeros((nmode, nmode))
	    for i in range(nmode):
		    width_matrix[i][i] = (fred*np.sqrt(width*reference_frequencies[i]))**2

    return width_matrix





