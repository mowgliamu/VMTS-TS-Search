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

from copy import deepcopy
from libspec import path_exists

from alignment import *
from input_parameters import *
from physical_constants import *
from gaussian_interface import *


# Print Precision!
np.set_printoptions(precision=8, suppress=True)

#sys.tracebacklimit=0


def get_x_to_q_transformation_matrix(param_list, write_to_file):

    '''
    Transformation matrices for X -> Q and Q -> X transformation.
    Simple, straightforward. No Eckart business here. 

    natom:	param_list[0]
    amass:	param_list[1]
    nmode:	param_list[2]
    ref_cart:	param_list[3]
    ref_freq:	param_list[4]
    L_ref_mat:	param_list[5]

    -------------
    MASS-WEIGHTED
    -------------

    Q -> X: M^(-0.5) * L_ref_mat ("D_mat")
    X -> Q = (D.T * D)^-1 * D.T  ("D_dagger_mat")

    -------------
    DIMENSIONLESS
    -------------

    '''

    # Open Zip!
    natom	=	param_list[0]
    amass	=	param_list[1]
    nmode	=	param_list[2]
    ref_cart	=	param_list[3]
    ref_freq	=	param_list[4]
    L_ref_mat	=	param_list[5]

    # Get the matrix of atomic masses 
    mass_matrix = np.diag(np.repeat(amass, 3))
    mass_matrix_sqrt = np.diag(np.repeat(np.sqrt(amass), 3))
    mass_matrix_sqrt_div = np.diag(np.repeat(1.0/np.sqrt(amass), 3))

    #-------------
    #MASS-WEIGHTED
    #-------------

    # 1. Q -> X
    D_mat = np.dot(mass_matrix_sqrt_div, L_ref_mat)

    # 2. X -> Q
    DTDI = np.linalg.inv(np.dot(D_mat.T, D_mat))
    D_dagger_mat =  np.dot(DTDI, D_mat.T)

    #-------------
    #DIMENSIONLESS
    #-------------

    # 3. The Transformation Matrix : D~ [MCTDH Notation] (3N - 3, 3N)
    Qdmfs = np.zeros(np.shape(L_ref_mat)).T
    for alpha in range(nmode):
	for i in range(natom):	
	    for j in range(3):
		k = 3*i + j
		Qdmfs[alpha][k] = (fred*np.sqrt(ref_freq[alpha])*np.sqrt(amass[i]))*L_ref_mat.T[alpha][k]

    # 4. The Transformation Matrix: D~' [MCTDH Notation] (3N, 3N - 3)
    Xdmfs = np.zeros(np.shape(L_ref_mat))
    for alpha in range(nmode):
	for i in range(natom):	
	    for j in range(3):
		k = 3*i + j
		Xdmfs[k][alpha] = L_ref_mat[k][alpha]/(fred*np.sqrt(ref_freq[alpha])*np.sqrt(amass[i]))


    # ====================================
    # Write to file along with a test case
    # ====================================

    # Open file
    gwrite = open('transform_cartesian_normal', 'w')

    gwrite.write(str(natom)+'\n')
    for i in range(natom):
	gwrite.write("{:20.8f}".format(amass[i]))
    gwrite.write('\n')
    gwrite.write(str(nmode)+'\n')

    # Write reference geometry in cartesian coordinates
    gwrite.write('Cartesian Reference Geometry\n')
    for i in range(natom):
	for j in range(3):
	    gwrite.write("{:20.8f}".format(ref_cart[i][j]))
	gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write transformation matrix for Q --> X transformation
    gwrite.write('Transformation Matrix for Q to X\n')
    for i in range(3*natom):
	for j in range(nmode):
	    if not dimensionless:
		gwrite.write("{:20.8f}".format(D_mat[i][j]))
	    else:
		gwrite.write("{:20.8f}".format(Xdmfs[i][j]))

	gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write transformation matrix for X --> Q transformation
    gwrite.write('Transformation Matrix for X to Q\n')
    for i in range(nmode):
	for j in range(3*natom):
	    if not dimensionless:
		gwrite.write("{:20.8f}".format(D_dagger_mat[i][j]))
	    else:
		gwrite.write("{:20.8f}".format(Qdmfs[i][j]))
	gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write Reference Normal Modes
    gwrite.write('Reference Normal Modes\n')
    for i in range(3*natom):
	for j in range(nmode):
		gwrite.write("{:20.8f}".format(L_ref_mat[i][j]))
	gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write reference frequencies
    gwrite.write('Reference Frequencies\n')
    for i in range(nmode):
	gwrite.write("{:20.8f}".format(ref_freq[i]))
    gwrite.write('\n')
    gwrite.write('\n')
     
    gwrite.close()

    #  Finally Return
    if not dimensionless:
	return D_mat, D_dagger_mat
    else:
	return Xdmfs, Qdmfs

