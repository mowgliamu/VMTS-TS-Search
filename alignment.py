# -*- coding: utf-8 -*-
#!usr/bin/env python


# A script to drive calculation for ts search using VIBRON.

import re
import os
import sys
import numpy as np
from subprocess import call
from libspec import path_exists


def get_mwcart(natom, amass, input_cart):

    '''
    Mass-weighted Cartesian coordinates
    '''
    
    mwcart = np.zeros((natom, 3))

    for i in range(natom):
	mwcart[i][0] = np.sqrt(amass[i])*input_cart[i][0]
	mwcart[i][1] = np.sqrt(amass[i])*input_cart[i][1]
	mwcart[i][2] = np.sqrt(amass[i])*input_cart[i][2]

    return mwcart

def get_centre_of_mass(natom, amass, input_cart):

    '''
    Centre of Mass using m.w. Cartesian coordinates
    '''

    TotMass    = np.sum(amass)
    CentreMass = np.zeros(3)

    #mwcart =  get_mwcart(natom, amass, input_cart)

    for i in range(natom):
	CentreMass[0] += amass[i]*input_cart[i][0]
	CentreMass[1] += amass[i]*input_cart[i][1]
	CentreMass[2] += amass[i]*input_cart[i][2]

    CentreMass = CentreMass/TotMass

    return CentreMass


def get_inertia_tensor(natom, amass, input_cart):

    '''
    Inertia Tensor (3x3 Matrix) using m.w. Cartesian coordinates
    '''

    Inertia_Tensor = np.zeros((3,3))

    mwcart =  get_mwcart(natom, amass, input_cart)

    for i in range(natom):
	Inertia_Tensor[0][0] += mwcart[i][1]*mwcart[i][1] + mwcart[i][2]*mwcart[i][2]
	Inertia_Tensor[1][1] += mwcart[i][0]*mwcart[i][0] + mwcart[i][2]*mwcart[i][2]
	Inertia_Tensor[2][2] += mwcart[i][0]*mwcart[i][0] + mwcart[i][1]*mwcart[i][1]
	Inertia_Tensor[0][1] += mwcart[i][0]*mwcart[i][1]
	Inertia_Tensor[0][2] += mwcart[i][0]*mwcart[i][2]
	Inertia_Tensor[1][2] += mwcart[i][1]*mwcart[i][2]

    Inertia_Tensor[1][0] = Inertia_Tensor[0][1]
    Inertia_Tensor[2][0] = Inertia_Tensor[0][2]
    Inertia_Tensor[2][1] = Inertia_Tensor[1][2]

    return Inertia_Tensor



def get_distance_matrix(cartesian_geometry):

    '''
    Interatomic distances - all pairs
    '''

    # Distance Matrix [Interatomic distance for all pairs for a given geometry]
    ndistmat = natom*(natom + 1)/2
    distance_matrix = np.zeros(ndistmat)

    for i in range(natom):
	for j in range(i, natom):
	    pair_index = natom*i + j - ((i*(i+1))/2)
	    distance_matrix[pair_index] = euclidean_distance(cartesian_geometry[i,:],\
							     cartesian_geometry[j,:])

    return distance_matrix


def eckart_rotation_general(natom, amass, target_cart, ref_cart):

    '''
    Satisfying Eckart conditions as an RMSD problem, solve using Quaternions.
    '''

    # Initialize
    C_matrix = np.zeros((4,4))
    Eckart_rotation_matrix = np.zeros((3,3))

    displacement_plus = np.zeros((natom, 3))
    displacement_mins = np.zeros((natom, 3))

    # Get mass-weighted cartesian
    ref_mwcart = get_mwcart(natom, amass, ref_cart)
    tar_mwcart = get_mwcart(natom, amass, target_cart)

    # Displacement Vector
    for i in range(natom):
	displacement_plus[i,:] = ref_mwcart[i,:] + tar_mwcart[i,:]
	displacement_mins[i,:] = ref_mwcart[i,:] - tar_mwcart[i,:]

    # C Matrix
    for i in range(natom):
	C_matrix[0][0] += displacement_mins[i][0]**2 + displacement_mins[i][1]**2 + displacement_mins[i][2]**2
	C_matrix[1][1] += displacement_mins[i][0]**2 + displacement_plus[i][1]**2 + displacement_plus[i][2]**2
	C_matrix[2][2] += displacement_plus[i][0]**2 + displacement_mins[i][1]**2 + displacement_plus[i][2]**2
	C_matrix[3][3] += displacement_plus[i][0]**2 + displacement_plus[i][1]**2 + displacement_mins[i][2]**2
	C_matrix[0][1] += displacement_plus[i][1]*displacement_mins[i][2] - displacement_mins[i][1]*displacement_plus[i][2]
	C_matrix[0][2] += displacement_mins[i][0]*displacement_plus[i][2] - displacement_plus[i][0]*displacement_mins[i][2]
	C_matrix[0][3] += displacement_plus[i][0]*displacement_mins[i][1] - displacement_mins[i][0]*displacement_plus[i][1]
	C_matrix[1][2] += displacement_mins[i][0]*displacement_mins[i][1] - displacement_plus[i][0]*displacement_plus[i][1]
	C_matrix[1][3] += displacement_mins[i][0]*displacement_mins[i][2] - displacement_plus[i][0]*displacement_plus[i][2]
	C_matrix[2][3] += displacement_mins[i][1]*displacement_mins[i][2] - displacement_plus[i][1]*displacement_plus[i][2]

    C_matrix[1][0] = C_matrix[0][1]
    C_matrix[2][0] = C_matrix[0][2]
    C_matrix[3][0] = C_matrix[0][3]
    C_matrix[2][1] = C_matrix[1][2]
    C_matrix[3][1] = C_matrix[1][3]
    C_matrix[3][2] = C_matrix[2][3]

    # Diagonalize C Matrix	
    cval, cvec = np.linalg.eigh(C_matrix)

    # Select eigevector for lowest eigenvalue
    quaternion_rotation = cvec[:,0]		#A Quanternion object!

    # Build Eckart rotation matrix from this quaternion
    Eckart_rotation_matrix[0][0] = quaternion_rotation[0]**2 + quaternion_rotation[1]**2 - quaternion_rotation[2]**2 - quaternion_rotation[3]**2
    Eckart_rotation_matrix[1][1] = quaternion_rotation[0]**2 - quaternion_rotation[1]**2 + quaternion_rotation[2]**2 - quaternion_rotation[3]**2
    Eckart_rotation_matrix[2][2] = quaternion_rotation[0]**2 - quaternion_rotation[1]**2 - quaternion_rotation[2]**2 + quaternion_rotation[3]**2
    Eckart_rotation_matrix[0][1] = 2.0*(quaternion_rotation[1]*quaternion_rotation[2] + quaternion_rotation[0]*quaternion_rotation[3])
    Eckart_rotation_matrix[1][0] = 2.0*(quaternion_rotation[1]*quaternion_rotation[2] - quaternion_rotation[0]*quaternion_rotation[3])
    Eckart_rotation_matrix[0][2] = 2.0*(quaternion_rotation[1]*quaternion_rotation[3] - quaternion_rotation[0]*quaternion_rotation[2])
    Eckart_rotation_matrix[2][0] = 2.0*(quaternion_rotation[1]*quaternion_rotation[3] + quaternion_rotation[0]*quaternion_rotation[2])
    Eckart_rotation_matrix[1][2] = 2.0*(quaternion_rotation[2]*quaternion_rotation[3] + quaternion_rotation[0]*quaternion_rotation[1])
    Eckart_rotation_matrix[2][1] = 2.0*(quaternion_rotation[2]*quaternion_rotation[3] - quaternion_rotation[0]*quaternion_rotation[1])

    # 3N x 3N "B matrix" with diagonal blocks as T matrix!
    B_matrix = np.kron(np.eye(natom), Eckart_rotation_matrix)	

    # calculate residual
    residual_mass = 0.0
    residual_no_mass = 0.0
    for n in range(natom):
	residual_no_mass += (np.linalg.norm(target_cart[n,:] - (np.dot(Eckart_rotation_matrix, ref_cart[n,:]))))**2
	residual_mass += (np.linalg.norm(tar_mwcart[n,:] - (np.dot(Eckart_rotation_matrix, ref_mwcart[n,:]))))**2


    return Eckart_rotation_matrix, B_matrix


def check_eckart_condition(natom, amass, inp_cart, ref_cart):

    '''
    Build A matrix, and check if it is symmetric.
    If yes, Eckart conditions are satisfied. 
    '''

    A_mat = np.zeros((3, 3))

    for i in range(natom):
	A_mat[0][0] += amass[i]*ref_cart[i][0]*inp_cart[i][0]
	A_mat[0][1] += amass[i]*ref_cart[i][0]*inp_cart[i][1]
	A_mat[0][2] += amass[i]*ref_cart[i][0]*inp_cart[i][2]
	A_mat[1][0] += amass[i]*ref_cart[i][1]*inp_cart[i][0]
	A_mat[1][1] += amass[i]*ref_cart[i][1]*inp_cart[i][1]
	A_mat[1][2] += amass[i]*ref_cart[i][1]*inp_cart[i][2]
	A_mat[2][0] += amass[i]*ref_cart[i][2]*inp_cart[i][0]
	A_mat[2][1] += amass[i]*ref_cart[i][2]*inp_cart[i][1]
	A_mat[2][2] += amass[i]*ref_cart[i][2]*inp_cart[i][2]

    return A_mat


def project_out_translation_rotation(natom, amass, input_cart):

    '''
    Return Projection Operator for Translation/Rotation
    '''

    # Get Centre of Mass and Moment of Inertia
    com = get_centre_of_mass(natom, amass, input_cart)
    moi = get_inertia_tensor(natom, amass, input_cart)

    # Initialize (3N, 6) array for Translation and Rotation
    Dmat = np.zeros((3*natom, 6), dtype=float)

    # Construct Eigenvectors correspoding to Translation
    for i in range(3):
	for k in range(natom):
	    for alpha in range(3):
		if alpha == i:
		    Dmat[3*k+alpha, i] = np.sqrt(amass[k])
		else:
		    pass


    xyzcom = input_cart - com

    # Construct Eigenvectors correspoding to Rotation
    # 1. Get Inertia Tensor and Diagonalize
    Ival, Ivec = np.linalg.eigh(moi)
    
    # 2. Construct Pmat
    Pmat = np.dot(xyzcom, Ivec)
    
    # 3. Construct Rotational Normal Coordinates
    for i in range(natom):
	for j in range(3):
	    Dmat[3*i + j, 3] = (Pmat[i, 1]*Ivec[j, 2] - Pmat[i, 2]*Ivec[j, 1])*np.sqrt(amass[i])
	    Dmat[3*i + j, 4] = (Pmat[i, 2]*Ivec[j, 0] - Pmat[i, 0]*Ivec[j, 2])*np.sqrt(amass[i])
	    Dmat[3*i + j, 5] = (Pmat[i, 0]*Ivec[j, 1] - Pmat[i, 1]*Ivec[j, 0])*np.sqrt(amass[i])

    
    # Orthonormalize
    LTR, xxx = np.linalg.qr(Dmat)

    # Get Projection operator
    Projection = np.dot(LTR, LTR.T)

    return LTR, Projection


