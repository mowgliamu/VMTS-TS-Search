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

class OptimizeAlphaBeta(object):

    """
    A class to optimize alpha and beta parameters to obtain the best IRC.
    """

    def __init__(self):

	self.all_irc_data = None


    def map_window_parameters(self, opt_params, opt_pointers, pot_window):

	'''
	Map all (alpha, beta) to appropriate places in pot_window.
	'''

	m = 0
	for key in opt_pointers.keys():
	    nparam = len(opt_pointers[key])
	    if nparam != 0:
		for i in range(nparam):
		    pot_window[key][opt_pointers[key][i]] = opt_params[m]
		    m=m+1
		    
	return pot_window


    def get_objective_function(self, pot_minima, pot_tstate, pot_window, flag_derivative):

	'''
	Simply evaluate E = sum_i (E_i_abinitio - E_i_model)**2
	Along all the IRC points (i) / Or at a given set of geometries.
	'''
	
	# Optimize model parameters for current guess and calculate IRC [TS only]
	if not flag_derivative:
	    if optimize_alpha_beta == 'ts_only':
		call_optimize_ts(pot_minima, pot_tstate, pot_window, 1, ab_to_alpha_map, window_flag)
		#driver_optimize_ts_vibronic_parameters(nconfig, pot_minima, pot_tstate, pot_window,\
			#	   					ab_to_alpha_map, 1e-3, window_flag)
	    elif optimize_alpha_beta == 'full':
		driver_optimize_all_vibronic_parameters(nconfig, pot_minima, pot_tstate, pot_window,\
		    					ab_to_alpha_map, 1e-4, window_flag)
	    else:
		pass
	else:
	    pass

	# Perform IRC on Current Model
	vm_all_data = [pot_minima, pot_tstate, pot_window]
	vm_irc_energies, vm_irc_gnorm = call_irc_objective_function(vm_all_data)

	# Sum of difference squares - Gaussian vs Vibronic Model
	if not include_grad_norm:
	    E_current = np.sum((gaussian_energies - vm_irc_energies)**2.0)
	else:
	    E_current = np.sum((gaussian_energies - vm_irc_energies)**2.0) + np.sum((gaussian_gradnorm - vm_irc_gnorm)**2.0)

	return E_current


    def get_function_and_gradient(self, all_params, opt_params, pot_minima, pot_tstate, pot_window, epsilon):

	'''
	'''
    
	self.map_window_parameters(all_params, opt_params, pot_window)

	print
	print '@get_function_gradient_hessian: pot_window'
	print pot_window
	print

	# Current function value: This call mutates pot_tstate!
	flag_opt_derivative = False
	E_current = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

	print
	print '@get_function_and_gradient: E_current'
	print E_current
	print

	# Total number of parameters that are being optimized
	nparam = 0
	all_values = opt_params.values()
	for p in range(len(all_values)):
	    nparam += len(all_values[p]) 
	
	# Initialize gradient and hessian
	current_gradient = np.zeros(nparam)
	current_hessian = np.zeros((nparam, nparam))

	# Change flag so as to NOT do any optimization of vibronic model for gradient calculation
	#flag_opt_derivative = True
	flag_opt_derivative = False

	for i in range(nparam):

	    # Plus displacement
	    all_params[i] = all_params[i] + epsilon
	    pot_window = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
	    E_plus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

	    # Minus displacement
	    all_params[i] = all_params[i] - 2.0*epsilon
	    pot_window = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
	    E_minus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

	    current_gradient[i] = (E_plus - E_minus)/(2.0*epsilon)

	    # RESET
	    all_params[i] = all_params[i] + epsilon

	# Examine parameters that need to be optimized
	print
	print '@get_function_and_gradient: Gradient val'
	print current_gradient
	print

	return E_current, current_gradient


    def get_hessian(self, all_params, opt_params, pot_minima, pot_tstate, pot_window, epsilon):

	'''
	'''
    
	self.map_window_parameters(all_params, opt_params, pot_window)

	print
	print '@get_function_gradient_hessian: pot_window'
	print pot_window
	print

	# Current function value: This call mutates pot_tstate!
	flag_opt_derivative = False
	E_current = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

	# Total number of parameters that are being optimized
	nparam = 0
	all_values = opt_params.values()
	for p in range(len(all_values)):
	    nparam += len(all_values[p]) 
	
	# Initialize gradient and hessian
	current_hessian = np.zeros((nparam, nparam))

	# Set flag so as to NOT do any optimization of vibronic model for gradient calculation
	flag_opt_derivative = True


	for i in range(nparam):

	    # Plus displacement
	    all_params[i] = all_params[i] + epsilon
	    pot_window = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
	    E_plus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

	    # Minus displacement
	    all_params[i] = all_params[i] - 2.0*epsilon
	    pot_window = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
	    E_minus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

	    # RESET
	    all_params[i] = all_params[i] + epsilon

	    for j in range(nparam):
		if i == j:
		    current_hessian[i][j] = (E_plus - 2.0*E_current + E_minus)/(epsilon**2.0)
		else:

		    # Plus Plus displacement
		    all_params[i] = all_params[i] + epsilon
		    all_params[j] = all_params[j] + epsilon
		    pot_window  = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
		    E_plus_plus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

		    # Plus Minus displacement
		    all_params[j] = all_params[j] - 2.0*epsilon
		    pot_window  = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
		    E_plus_minus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

		    # Minus Plus displacement
		    all_params[i] = all_params[i] - 2.0*epsilon
		    all_params[j] = all_params[j] + 2.0*epsilon
		    pot_window  = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
		    E_minus_plus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

		    # Minus Minus displacement
		    all_params[j] = all_params[j] - 2.0*epsilon
		    pot_window  = self.map_window_parameters(all_params, opt_params, pot_window)	# Mutation [Return is only for clarity]
		    E_minus_minus = self.get_objective_function(pot_minima, pot_tstate, pot_window, flag_opt_derivative)

		    current_hessian[i][j] = (E_plus_plus - E_plus_minus - E_minus_plus + E_minus_minus)/(4.0*(epsilon)**2.0)

		    # RESET
		    all_params[i] = all_params[i] + 2.0*epsilon
		    all_params[j] = all_params[j] + 2.0*epsilon


	# Examine parameters that need to be optimized
	print
	print '@get_function_and_gradient: Hessian eigenvalues'
	hval, hvec = np.linalg.eigh(current_hessian)
	print hval
	print

	return current_hessian



    def optimization_oned_newton(self, params, opt_param, pot_minima, pot_tstate, pot_window, epsilon, max_step):

	'''
	Newton-Raphson 1D code.

	NOTE: Make convergence criteira loose! [Small changes in Function / Gradient / Parameter]
	'''

	# Iterations 
	max_iter = 60
	err_tol  = 1e-4
	
	# Set max_step for Newton
	#max_step = 0.1

	# Zip IN
	#key = opt_param.keys()[0]
	#val_index = opt_param.values()[0][0]
	#params = np.array([pot_window[key][val_index]])

	# Keep Track of Gradient Sign Change
	last_gradient_sign = 1
	last_hessian_sign = 1
	gradient_sign_changes = 0

	next_hessian = 0.0
	last_parameter = params[0]
	
	flag = True

	for i in range(1, max_iter+1):

	    # Set params_new = 0.0
	    params_new = np.array([0.0])

	    print 
	    print '@optimization_oned_newton: Current Newton Cycle -', i
	    print
	    print '@optimization_oned_newton: Current Guess -', params[0]
	    print

	    # Update pot_window
	    #pot_window[key][val_index] = params[0]

	    # Get Function, Gradient and Hessian for Current Guess
	    E_current, E_gradient = self.get_function_and_gradient(params, opt_param, pot_minima, pot_tstate, pot_window, epsilon)
	    E_hessian = self.get_hessian(params, opt_param, pot_minima, pot_tstate, pot_window, epsilon)
	    E_gradient = np.copy(E_gradient[0])
	    E_hessian  = np.copy(E_hessian[0][0])

	    print
	    print 'Function value in Cycle', i
	    print E_current
	    print
	    print 'Gradient value in Cycle', i
	    print E_gradient
	    print
	    print 'Hessian Matrix in Cycle:', i
	    print E_hessian
	    print

	    # Store previous
	    last_gradient = E_gradient
	    last_hessian = E_hessian
	    
	    # Keep track of Sign change of gradient
	    current_gradient_sign = E_gradient/abs(E_gradient)
	    if current_gradient_sign == -last_gradient_sign:
		gradient_sign_changes += 1
		last_gradient_sign = current_gradient_sign
	    else:
		pass

	    print
	    print 'Sign changes of Gradients in Cycle:', i
	    print gradient_sign_changes
	    print

	    # Bracket search and newton step  starts here!
	    if gradient_sign_changes >= 1 and np.sign(last_hessian) > 0 and np.sign(next_hessian) > 0:
		print
		print 'Gradient sign change for +ve hessian found. A minimum has been crossed!'
		print

		# Set bracket: min and max based on gradient for the first time
		if flag: 
		    if last_gradient < 0.0:
			min_param = next_parameter
			max_param = last_parameter
		    elif last_gradient > 0.0:
			min_param = last_parameter
			max_param = next_parameter
		    else:
			pass
		else:
		    pass

		print
		print 'Initial Bracket - min, max:', min_param, max_param
		print

		# Get mid point of min and max!
		mid_point = (min_param + max_param)/2.0

		# Update pot_window
		#pot_window[key][val_index] = mid_point

		# Get Function and Gradient at this mid point
		E_current, E_gradient = self.get_function_and_gradient(np.array([mid_point]), opt_param, pot_minima, pot_tstate, pot_window, epsilon)
		E_hessian = self.get_hessian(np.array([mid_point]), opt_param, pot_minima, pot_tstate, pot_window, epsilon)
		E_gradient = np.copy(E_gradient[0])
		E_hessian  = np.copy(E_hessian[0][0])

		# Take the newton step at mid point and get new guess
		newton_step = E_gradient/E_hessian
		new_guess = params[0] - newton_step

		print
		print 'New point with Newton step inside bisection search - mid point'
		print new_guess
		print

		# Check if Newton step is inside the Bracket
		# If yes, make it the new guess. If not, make the mid point as the new guess

		if min_param < new_guess < max_param:
		    next_step = newton_step
		    params_new[0] = params[0] - next_step
		else:
		    new_guess = mid_point
		    params_new[0] = mid_point

		# Update pot_window
		#pot_window[key][val_index] = params_new[0]

		# Calculate Gradient again at new guess to set new bracket
		E_current, E_gradient = self.get_function_and_gradient(params_new, opt_param, pot_minima, pot_tstate, pot_window, epsilon)
		E_hessian = self.get_hessian(params_new, opt_param, pot_minima, pot_tstate, pot_window, epsilon)
		E_gradient = np.copy(E_gradient[0])
		E_hessian  = np.copy(E_hessian[0][0])

		# Set new boundaries (oh, boundaries!)
		if E_gradient < 0.0:
		    min_param = new_guess
		else:
		    max_param = new_guess

		flag = False

		print
		print 'Final Bracket - min, max:', min_param, max_param
		print

	    else: 
		# Reset gradient sign counter
		gradient_sign_changes = 0

		# Perform Regular Newton
		if E_hessian > 0.0:
		    print
		    print 'Taking the Newton step'
		    print
		    next_step = E_gradient/E_hessian
		    if abs(next_step) > max_step:
			print
			print 'Taking max step for +ve Hessian'
			print
			next_step = np.sign(next_step)*max_step
			params_new[0] = params[0] - next_step
			print
			print 'Indside Newton branch - check guesss (Params)'
			print params
			print
		    else:
			pass
			params_new[0] = params[0] - next_step
		elif E_hessian < 0.0:
		    print
		    print 'Taking max step for -ve Hessian'
		    print
		    next_step = np.sign(E_gradient)*max_step
		    params_new[0] = params[0] - next_step
		else:
		    print 'No step taken. Something might have gone wrong.'


	    # New parameter
	    print
	    print 'step in current cycle', next_step
	    print

	    print
	    print 'New parameters'
	    print params_new
	    print

	    print
	    print 'Last guess parameters'
	    print params
	    print

	    # Update pot_window
	    #pot_window[key][val_index] = params[0]

	    # Calculate Gradient again at new guess to set new bracket
	    E_current, E_gradient = self.get_function_and_gradient(params, opt_param, pot_minima, pot_tstate, pot_window, epsilon)
	    E_hessian = self.get_hessian(params, opt_param, pot_minima, pot_tstate, pot_window, epsilon)
	    E_gradient = np.copy(E_gradient[0])
	    E_hessian  = np.copy(E_hessian[0][0])

	    last_parameter = params[0]
	    next_parameter = params_new[0]
	    next_gradient  = E_gradient
	    next_hessian   = E_hessian

	    # Check convergence NOTE: UPDATE CONVERGENCE CRITERIA
	    if abs(E_gradient) < 0.02 and E_hessian > 0.0 and i<= max_iter:
		print
		print 'Covergence reached based on gradient and hessian after', i, 'cycles'
		print 'Value of alpha, beta = ', params_new
		print
		break
	    elif np.all(abs(params_new - params) < err_tol) and i<= max_iter:
		print
		print 'Covergence reached based on change in parameter value after', i, 'cycles'
		print 'Value of alpha, beta = ', params_new
		print
		break
	    elif np.any(abs(params_new - params) > err_tol) and i<max_iter:
		params = params_new
		continue
	    else:
		print 'Convergence not reached after', i, 'cycles'


	return params_new


    def get_optimal_alpha_beta_newton(self, params, opt_param, pot_minima, pot_tstate, pot_window, epsilon):

	'''
	My own implementation of Newton Algorithm to optimize alpha and beta together.
	'''

	# Iterations 
	max_iter = 50
	err_tol  = 1e-4
	
	# Set max_step for Newton
	max_step = 0.01

	for i in range(1, max_iter+1):

	    print 
	    print 'Current Newton Cycle:', i
	    print
	    print 'Current Guess Parameters:', params
	    print

	    # Get Function and Gradient for Current Guess
	    E_current, E_gradient = self.get_function_and_gradient(params, opt_param, pot_minima, pot_tstate, pot_window, epsilon)

	    # Get Hessian for Current Guess
	    E_hessian = self.get_hessian(params, opt_param, pot_minima, pot_tstate, pot_window, epsilon)

	    print
	    print 'Function value in Cycle', i
	    print E_current
	    print

	    print
	    print 'Gradient value in Cycle', i
	    print E_gradient
	    print

	    print
	    print 'Hessian Matrix in Cycle:', i
	    print E_hessian
	    print

	    # Diagonalize Hessian, Inspect Eigenvalues
	    hval, hvec = np.linalg.eigh(E_hessian)

	    print
	    print 'Hessian Eigenvalues in Cycle:', i
	    print hval
	    print

	    print
	    print 'Hessian Eigenvectors in Cycle:', i
	    print hvec
	    print

	    # Transform Gradient to Hessian Eigenvector Basis
	    gradient_transformed = np.dot(hvec.T, E_gradient)

	    next_step = np.zeros(2)
	    # Loop over all Hessian Eigenvalues
	    for m in range(2):
		if hval[m] > 0.0:
		    print
		    print 'Eigenvalue:', hval[m], 'Taking the Newton step'
		    print
		    next_step[m] = gradient_transformed[m]/hval[m]
		    if abs(next_step[m]) > max_step:
			print
			print 'Eigenvalue:', hval[m], 'Taking max step'
			print
			next_step[m] = np.sign(next_step[m])*max_step
		    else:
			pass
		elif hval[m] < 0.0:
		    print
		    print 'Eigenvalue:', hval[m], 'Taking max step'
		    print
		    next_step[m] = np.sign(gradient_transformed[m])*max_step
		else:
		    print 'No step taken. Something might have gone wrong.'
		    

	    # Transform step back to original basis
	    step = np.dot(hvec, next_step)
	    print
	    print 'step in current cycle', step
	    print

	    params_new = params - step

	    print
	    print 'New parameters'
	    print params_new
	    print

	    # Check convergence
	    if np.all(abs(params_new - params) < err_tol) and i<= max_iter:
		print
		print 'Covergence reached after', i, 'cycles'
		print 'Value of alpha, beta = ', params_new
		print
		break
	    elif np.any(abs(params_new - params) > err_tol) and i<max_iter:
		params = params_new
		continue
	    else:
		print 'Convergence not reached after', i, 'cycles'


	return params_new


    def get_optimal_alpha_beta_newton_scipy(self,  all_params, opt_params, pot_minima, pot_tstate, pot_window, epsilon):

	'''
	'''
	
	# Initial Guess
	#x0 = np.array((alpha_guess, beta_guess))

	# Minimize alpha, beta using Newton Conjugate-Gradient Method of Scipy Optimization
	res = minimize(self.get_function_and_gradient, all_params, args=(opt_params, pot_minima, pot_tstate, pot_window, epsilon),\
		       method='Newton-CG', jac=True, hess=self.get_hessian,\
		       options={'xtol': 1e-4, 'maxiter': 200, 'disp': True})

	return res




