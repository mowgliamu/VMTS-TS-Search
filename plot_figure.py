# -*- coding: utf-8 -*-
#!usr/bin/env python

import numpy as np
import cmath, math
from scipy import linalg
from pprint import pprint
from subprocess import call

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

from copy import deepcopy
from libspec import path_exists

# Print Precision!
np.set_printoptions(precision=8, suppress=True)


###############################
# Helper function for plotting
###############################

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename):
    plt.tight_layout(.4) # Ensures that everything fits http://matplotlib.org/users/tight_layout_guide.html
    #plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))

def set_x_axis_label(ax, data_type):

    if data_type == 'irc':
	ax.set_xlabel('Intrinsic Reaction Coordinate')
    elif data_type == 'ctp':
	ax.set_xlabel('Cubic Transit Path')
    elif data_type == 'qtp':
	ax.set_xlabel('Quadratic Transit Path')
    else:
	pass

    return

# LaTeX Stuff for Plot
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,                    # UserWarning: text.fontsize is deprecated and replaced with font.size
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }

#matplotlib.rcParams.update(pgf_with_latex)

