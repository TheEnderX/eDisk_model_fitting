#!/usr/bin/env python3

from pdspy.constants.astronomy import arcsec
import pdspy.interferometry as uv
import pdspy.imaging as im
import pdspy.misc as misc
import dynesty.plotting as dyplot
import dynesty.results as dyres
import dynesty.utils as dyfunc
import dynesty
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import scipy.signal
import schwimmbad
import itertools
import argparse
import corner
import numpy
import sys
import os
from mpi4py import MPI
import gc
import glob
import random

comm = MPI.COMM_WORLD

import pkg_resources
pdspy_v2 = int(pkg_resources.get_distribution("pdspy").version.split(".")[0]) \
        >= 2

# location of uv fits files parent directory.  Each EB must have a directory here named something.vis.
#               Inside these each fits file has one spw.  Don't forget to add / at end of directory
visfilesdir = '/scratch/yiweix3/eDisk/Ced110IRS4/UVDATA/'
imagefile = '/scratch/yiweix3/eDisk/Ced110IRS4/Ced110IRS4_SBLB_continuum_robust_0.0.image.tt0.fits'
# calculate by finding weight scale by 1./vis.weights.sum()**0.5, then matching that to the image RMS
# image RMS = 17.5 uJy/beam, and weights give
weightscale =  8.0
sourcename = 'Ced110IRS4'
#%%
################################################################################
#
# Set up all of the different models.
#
################################################################################

# The base, exponentially tapered model.
def exponentially_tapered_powerlaw_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)

    return params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma"]))

def exponentially_tapered_powerlaw2_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)

    return params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"]))

def exponentially_tapered_powerlaw2_ring_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]

    return (params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"])) * \
            (1. + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4))))))

def exponentially_tapered_powerlaw2_ring2_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]
    rp3 = rp - params["r_c3"]

    return (params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"])) * \
            (1. + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) \
            + params["b"] * (numpy.exp(-(rp3**4 / (2 * params["r_w3"]**4))))))

def asymmetric_exponentially_tapered_powerlaw2_ring_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]
    phi = numpy.arctan2(yi,xp)
    
    azimuthal_gap1 = numpy.exp(-(phi - params["phi_c"])**4 / (2 * params["phi_w"]**4))
    azimuthal_gap2 = numpy.exp(-(phi - (params["phi_c"] + 2 * numpy.pi))**4 / (2 * params["phi_w"]**4))
    azimuthal_gap3 = numpy.exp(-(phi - (params["phi_c"] - 2 * numpy.pi))**4 / (2 * params["phi_w"]**4))
    azimuthal_gap = numpy.maximum(numpy.maximum(azimuthal_gap1, azimuthal_gap2), azimuthal_gap3)
    
    return params["flux"] * \
            ((numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"]))) * \
            (1 + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) * azimuthal_gap)  

def asymmetric_exponentially_tapered_powerlaw2_sin_ring_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]
    phi = numpy.arctan2(yi,xp)

    return params["flux"] * \
            ((numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"]))) * \
            (1 + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) * \
            numpy.sin(phi - params["phi_o"]))   

def exponentially_tapered_powerlaw2_1spiral_dphi_model(params, xp, yp):
    params["phi_0"]=(params["phi_0"] + numpy.pi) % (2 * numpy.pi) - numpy.pi
    yi = yp / numpy.cos(params["incl"])

    rp = numpy.sqrt(xp**2 + yi**2)
    phi = numpy.arctan2(xp,yi)
    phi_array = numpy.array((phi-10.0*numpy.pi, phi-8.0*numpy.pi, phi-6.0*numpy.pi, phi-4.0*numpy.pi, phi-2.0*numpy.pi, \
    			phi, phi+2.0*numpy.pi, phi+4.0*numpy.pi, phi+6.0*numpy.pi, phi+8.0*numpy.pi, phi+10.0*numpy.pi))
    nphi_arr = len(phi_array)

    azimuthal_I=params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"]))
    
    azimuthal_I_sp=params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w_sp"])**(-params["gamma_sp"]) * numpy.exp(-(rp / params["r_w_sp"])**(2 - params["gamma2_sp"]))

    r_spiral_1=numpy.zeros(phi_array.shape)

    for i in range(nphi_arr):
       r_spiral_1[i]=params["r_w_sp"]/4.0 * numpy.exp(params["b_pitch"]*(phi_array[i]+params["phi_0"]))

    spiral_I_1=numpy.zeros(phi_array.shape)

    for i in range(nphi_arr):
       spiral_I_1[i]=azimuthal_I_sp * params["f_az"] * numpy.exp(-(rp-r_spiral_1[i])**2/2.0/(params["f_r"]*r_spiral_1[i])**2)


    azimuthal_I_spirals=azimuthal_I.copy()
    for i in range(nphi_arr):
       azimuthal_I_spirals+=spiral_I_1[i]

    return azimuthal_I_spirals

def exponentially_tapered_powerlaw2_1spiral_dphi_asym_ring_model(params, xp, yp):
    params["phi_0"]=(params["phi_0"] + numpy.pi) % (2 * numpy.pi) - numpy.pi
    yi = yp / numpy.cos(params["incl"])

    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]
    phi = numpy.arctan2(xp,yi)
    
    phi_array = numpy.array((phi-10.0*numpy.pi, phi-8.0*numpy.pi, phi-6.0*numpy.pi, phi-4.0*numpy.pi, phi-2.0*numpy.pi, \
    			phi, phi+2.0*numpy.pi, phi+4.0*numpy.pi, phi+6.0*numpy.pi, phi+8.0*numpy.pi, phi+10.0*numpy.pi))
    nphi_arr = len(phi_array)
    
    azimuthal_gap1 = numpy.exp(-(phi - params["phi_c"])**4 / (2 * params["phi_w"]**4))
    azimuthal_gap2 = numpy.exp(-(phi - (params["phi_c"] + 2 * numpy.pi))**4 / (2 * params["phi_w"]**4))
    azimuthal_gap3 = numpy.exp(-(phi - (params["phi_c"] - 2 * numpy.pi))**4 / (2 * params["phi_w"]**4))
    azimuthal_gap = numpy.maximum(numpy.maximum(azimuthal_gap1, azimuthal_gap2), azimuthal_gap3)

    azimuthal_I_as = params["flux"] * \
            	((numpy.sqrt(rp**2 + params["smooth"]**2) / \
            	params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"]))) * \
            	(1 + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) * azimuthal_gap)
    
    azimuthal_I_sp = params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            	params["r_w_sp"])**(-params["gamma_sp"]) * numpy.exp(-(rp / params["r_w_sp"])**(2 - params["gamma2_sp"]))

    r_spiral_1 = numpy.zeros(phi_array.shape)

    for i in range(nphi_arr):
       r_spiral_1[i] = params["r_w_sp"]/4.0 * numpy.exp(params["b_pitch"]*(phi_array[i]+params["phi_0"]))

    spiral_I_1 = numpy.zeros(phi_array.shape)

    for i in range(nphi_arr):
       spiral_I_1[i] = azimuthal_I_sp * params["f_az"] * numpy.exp(-(rp-r_spiral_1[i])**2/2.0/(params["f_r"]*r_spiral_1[i])**2)


    azimuthal_I = azimuthal_I_as.copy()
    for i in range(nphi_arr):
       azimuthal_I += spiral_I_1[i]

    return azimuthal_I

def exponentially_tapered_powerlaw2_ring_gaussian_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]

    return params["flux"] * \
	    ((numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"]))) * \
            (1. + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) + \
	    (params["c"] * numpy.exp(-rp**2 / (2 * params["r_w3"]**2))))

def exponentially_tapered_powerlaw2_ring2_gaussian_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]
    rp3 = rp - params["r_c3"]

    return params["flux"] * \
    	    ((numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma2"]))) * \
            (1. + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) \
            + params["b"] * (numpy.exp(-(rp3**4 / (2 * params["r_w3"]**4)))) \
            + params["c"] * numpy.exp(-rp**2 / ((2 * params["r_w4"]**2))))

def broken_powerlaw_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)

    return params["flux"] * (((numpy.sqrt(rp**2 + params["smooth"]**2)/params["r_w"])**(-params["gamma_in"])) * (0.5 * \
            (1. + ((numpy.sqrt(rp**2 \
            + params["smooth"]**2)/params["r_w"])**(1./params["delta"]))))**(-(params["gamma_out"]-params["gamma_in"])*params["delta"]))
            
def exponentially_tapered_broken_powerlaw_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)

    return params["flux"] * (((numpy.sqrt(rp**2 + params["smooth"]**2)/params["r_w"])**(-params["gamma_in"])) * (0.5 * \
            (1. + ((numpy.sqrt(rp**2 \
            + params["smooth"]**2)/params["r_w"])**(1./params["delta"]))))**(-(params["gamma_out"]-params["gamma_in"])*params["delta"])) \
            * numpy.exp(-(rp / params["r_exp"])**(2 - params["gamma2"]))

def exponentially_tapered_broken_powerlaw_asym_ring_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]
    phi = numpy.arctan2(yi,xp)

    azimuthal_gap1 = numpy.exp(-(phi - params["phi_c"])**4 / (2 * params["phi_w"]**4))
    azimuthal_gap2 = numpy.exp(-(phi - (params["phi_c"] + 2 * numpy.pi))**4 / (2 * params["phi_w"]**4))
    azimuthal_gap3 = numpy.exp(-(phi - (params["phi_c"] - 2 * numpy.pi))**4 / (2 * params["phi_w"]**4))
    azimuthal_gap = numpy.maximum(numpy.maximum(azimuthal_gap1, azimuthal_gap2), azimuthal_gap3)
    
    return params["flux"] * (((numpy.sqrt(rp**2 + params["smooth"]**2)/params["r_w"])**(-params["gamma_in"])) * (0.5 * \
            (1. + ((numpy.sqrt(rp**2 \
            + params["smooth"]**2)/params["r_w"])**(1./params["delta"]))))**(-(params["gamma_out"]-params["gamma_in"])*params["delta"])) \
            * numpy.exp(-(rp / params["r_exp"])**(2 - params["gamma2"])) * \
            (1 + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) * azimuthal_gap)

def exponentially_tapered_powerlaw_ring_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    phi = numpy.arctan2(yi,xp)

    return params["flux"] * ((numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * \
            numpy.exp(-(rp / params["r_w"])**(2 - params["gamma"])) + \
            params["f_a"] * (numpy.exp(-((rp - params["r_c"])**4 / (2 * params["r_r"]**4)))) * \
            numpy.exp(-(phi - params["phi_c"])**4) / (2 * params["phi_w"]**4))

def exponentially_tapered_powerlaw_gaussian4th_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]

    return (params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma"]))) * \
            (1.0 + params["a"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))))

def exponentially_tapered_gaussian4th_powerlaw_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)
    rp2 = rp - params["r_c"]

    return params["flux"] * (numpy.exp(-(rp2**4 / (2 * params["r_w2"]**4)))) * \
            (1.0 + params["a"] * ((numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * numpy.exp(-(rp / params["r_w"])**(2 - params["gamma"]))))


def exponentially_tapered_powerlaw_gap_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)

    return params["flux"] * (numpy.sqrt(rp**2 + params["smooth"]**2) / \
            params["r_w"])**(-params["gamma"]) * \
            numpy.exp(-(rp / params["r_w"])**(2 - params["gamma"])) - \
            params["flux_r"] * (numpy.exp(-((rp - params["r_c"])**4 / \
            (2 * params["r_w2"]**4)))) 

def gaussian_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2) 

    return params["flux"] * (numpy.exp(-rp**2 / ((2 * params["r_w"]**2))))

def gaussian_4th_ring_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2) 

    return params["flux"] * (numpy.exp(-((rp - params["r_c"])**4 / \
            (2 * params["r_w"]**4)))) \
            + params["flux_r"] * (numpy.exp(-((rp-params["r_r"])**4 / \
            (2 * params["r_w2"]**4))))

def gaussian_4th_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2) - params["r_c"]

    return params["flux"] * (numpy.exp(-(rp**4 / (2 * params["r_w"]**4))))

def rectangle_model(params, xp, yp):
    return params["flux"] * numpy.exp(-xp**4 / (2 * params["x_w"]**4) - \
            yp**4 / (2 * params["y_w"]**4))

def powerlaw_rectangle_model(params, xp, yp):
    return params["flux"] * (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_w"])**params["gamma"] * numpy.exp(-xp**4 / (2 * \
            params["x_w"]**4) - yp**4 / (2 * params["y_w"]**4))

def asymmetric_powerlaw_rectangle_model(params, xp, yp):
    return numpy.where(xp >= 0, params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["smooth"])**params["gamma1"] *\
            numpy.exp(-xp**4 / (2 * params["x_w"]**4) - yp**4 / (2 * \
            params["y_w"]**4)), params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["smooth"])**params["gamma2"] *\
            numpy.exp(-xp**4 / (2 * params["x_w"]**4) - yp**4 / (2 * \
            params["y_w"]**4)))

def asymmetric_powerlaw_rectangle_vartrunc_model(params, xp, yp):
    return numpy.where(xp >= 0, params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["smooth"])**params["gamma1"] *\
            numpy.exp(-numpy.abs(xp)**params["gammax"] / (2 * \
            params["x_w"]**params["gammax"]) - numpy.abs(yp)**params["gammay"] \
            / (2 * params["y_w"]**params["gammay"])), params["flux"] * \
            (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["smooth"])**params["gamma2"] * \
            numpy.exp(-numpy.abs(xp)**params["gammax"] / (2 * \
            params["x_w"]**params["gammax"]) - \
            numpy.abs(yp)**params["gammay"] / (2 * \
            params["y_w"]**params["gammay"])))

def flared_asymmetric_powerlaw_rectangle_vartrunc_model(params, xp, yp):
    y_w0 = params["y_w"] * (1 + params["scale"]*numpy.abs(xp) / params["x_w"])

    return numpy.where(xp >= 0, params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["smooth"])**params["gamma1"] *\
            numpy.exp(-numpy.abs(xp)**params["gammax"] / (2 * \
            params["x_w"]**params["gammax"]) - numpy.abs(yp)**params["gammay"]/\
            (2 * y_w0**params["gammay"])), params["flux"] * \
            (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["smooth"])**params["gamma2"] * \
            numpy.exp(-numpy.abs(xp)**params["gammax"] / (2 * \
            params["x_w"]**params["gammax"]) - numpy.abs(yp)**params["gammay"]/\
            (2 * y_w0**params["gammay"])))

def broken_powerlaw_rectangle_model(params, xp, yp):
    return params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["x_t"])**-params["gamma_in"] * \
            (0.5 * (1 + (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_t"])**(1./params["delta"])))**((params["gamma_in"] - \
            params["gamma_out"])*params["delta"]) * numpy.exp(-xp**4 / \
            (2 * params["x_w"]**4) - yp**4 / (2 * params["y_w"]**4))

def asymmetric_broken_powerlaw_rectangle_model(params, xp, yp):
    y_w0 = params["y_w"]

    return numpy.where(xp >= 0, params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["x_t"])**-params["gamma_in"] * \
            (0.5 * (1 + (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_t"])**(1./params["delta"])))**((params["gamma_in"] - \
            params["gamma_out1"])*params["delta"]) * numpy.exp(-numpy.abs(xp)**\
            4 / (2 * params["x_w"]**4) - \
            numpy.abs(yp)**4 / (2 * y_w0**4)), \
            params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["x_t"])**-params["gamma_in"] * \
            (0.5 * (1 + (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_t"])**(1./params["delta"])))**((params["gamma_in"] - \
            params["gamma_out2"])*params["delta"]) * numpy.exp(-numpy.abs(xp)**\
            4 / (2 * params["x_w"]**4) - \
            numpy.abs(yp)**4 / (2 * y_w0**4)))

def asymmetric_broken_powerlaw_rectangle_vartrunc_model(params, xp, yp):
    y_w0 = params["y_w"]

    return numpy.where(xp >= 0, params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["x_t"])**-params["gamma_in"] * \
            (0.5 * (1 + (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_t"])**(1./params["delta"])))**((params["gamma_in"] - \
            params["gamma_out1"])*params["delta"]) * numpy.exp(-numpy.abs(xp)**\
            params["gammax"] / (2 * params["x_w"]**params["gammax"]) - \
            numpy.abs(yp)**params["gammay"] / (2 * y_w0**params["gammay"])), \
            params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["x_t"])**-params["gamma_in"] * \
            (0.5 * (1 + (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_t"])**(1./params["delta"])))**((params["gamma_in"] - \
            params["gamma_out2"])*params["delta"]) * numpy.exp(-numpy.abs(xp)**\
            params["gammax"] / (2 * params["x_w"]**params["gammax"]) - \
            numpy.abs(yp)**params["gammay"] / (2 * y_w0**params["gammay"])))

def flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model(params, xp, yp):
    y_w0 = params["y_w"] * (1 + params["scale"]*numpy.abs(xp) / params["x_w"])

    return numpy.where(xp >= 0, params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["x_t"])**-params["gamma_in"] * \
            (0.5 * (1 + (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_t"])**(1./params["delta"])))**((params["gamma_in"] - \
            params["gamma_out1"])*params["delta"]) * numpy.exp(-numpy.abs(xp)**\
            params["gammax"] / (2 * params["x_w"]**params["gammax"]) - \
            numpy.abs(yp)**params["gammay"] / (2 * y_w0**params["gammay"])), \
            params["flux"] * (numpy.sqrt(xp**2 + \
            params["smooth"]**2) / params["x_t"])**-params["gamma_in"] * \
            (0.5 * (1 + (numpy.sqrt(xp**2 + params["smooth"]**2) / \
            params["x_t"])**(1./params["delta"])))**((params["gamma_in"] - \
            params["gamma_out2"])*params["delta"]) * numpy.exp(-numpy.abs(xp)**\
            params["gammax"] / (2 * params["x_w"]**params["gammax"]) - \
            numpy.abs(yp)**params["gammay"] / (2 * y_w0**params["gammay"])))

# Dictionaries that contain relevant info.

model_functions = {\
        "rectangle":rectangle_model,\
        "exponentially_tapered_powerlaw":exponentially_tapered_powerlaw_model,\
        "exponentially_tapered_powerlaw2":exponentially_tapered_powerlaw2_model,\
        "exponentially_tapered_powerlaw2_ring":exponentially_tapered_powerlaw2_ring_model,\
        "exponentially_tapered_powerlaw2_ring2":exponentially_tapered_powerlaw2_ring2_model,\
        "exponentially_tapered_powerlaw2_ring_gaussian":exponentially_tapered_powerlaw2_ring_gaussian_model,\
	"exponentially_tapered_powerlaw2_ring2_gaussian":exponentially_tapered_powerlaw2_ring2_gaussian_model,\
        "exponentially_tapered_powerlaw2_1spiral_dphi":exponentially_tapered_powerlaw2_1spiral_dphi_model,\
        "exponentially_tapered_powerlaw2_1spiral_dphi_asym_ring":exponentially_tapered_powerlaw2_1spiral_dphi_asym_ring_model,\
	"asymmetric_exponentially_tapered_powerlaw2_ring":asymmetric_exponentially_tapered_powerlaw2_ring_model,\
        "asymmetric_exponentially_tapered_powerlaw2_sin_ring":asymmetric_exponentially_tapered_powerlaw2_sin_ring_model,\
	"broken_powerlaw":broken_powerlaw_model, \
	"exponentially_tapered_broken_powerlaw":exponentially_tapered_broken_powerlaw_model, \
	"exponentially_tapered_broken_powerlaw_asym_ring":exponentially_tapered_broken_powerlaw_asym_ring_model, \
        "exponentially_tapered_powerlaw_gaussian4th":exponentially_tapered_powerlaw_gaussian4th_model, \
        "exponentially_tapered_gaussian4th_powerlaw": exponentially_tapered_gaussian4th_powerlaw_model, \
        "exponentially_tapered_powerlaw_ring":exponentially_tapered_powerlaw_ring_model,\
        "exponentially_tapered_powerlaw_gap":exponentially_tapered_powerlaw_gap_model,\
        "gaussian":gaussian_model,\
        "gaussian_4th":gaussian_4th_model,\
        "gaussian_4th_ring":gaussian_4th_ring_model,\
        "powerlaw_rectangle":powerlaw_rectangle_model,\
        "asymmetric_powerlaw_rectangle":asymmetric_powerlaw_rectangle_model,\
        "asymmetric_powerlaw_rectangle_vartrunc":\
                asymmetric_powerlaw_rectangle_vartrunc_model,\
        "flared_asymmetric_powerlaw_rectangle_vartrunc":\
                flared_asymmetric_powerlaw_rectangle_vartrunc_model,\
        "flared_asymmetric_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_powerlaw_rectangle_vartrunc_model,\
        "gapped_flared_asymmetric_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_powerlaw_rectangle_vartrunc_model,\
        "symmetric_gapped_flared_asymmetric_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_powerlaw_rectangle_vartrunc_model,\
        "broken_powerlaw_rectangle":\
                broken_powerlaw_rectangle_model,\
        "asymmetric_broken_powerlaw_rectangle":\
                asymmetric_broken_powerlaw_rectangle_model,\
        "asymmetric_broken_powerlaw_rectangle_vartrunc":\
                asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "flared_asymmetric_broken_powerlaw_rectangle_vartrunc":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "subgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "divgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "symmetric_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "symmetric_subgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        "symmetric_divgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":\
                flared_asymmetric_broken_powerlaw_rectangle_vartrunc_model,\
        }

names = {\
        "rectangle":["x0","y0","logx_w","logy_w","pa","logflux"],\
        "exponentially_tapered_powerlaw":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma"],\
        "exponentially_tapered_powerlaw2":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2"],\
        "exponentially_tapered_powerlaw2_ring":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "logr_c","logr_w2","a"],\
	"exponentially_tapered_powerlaw2_ring2":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "logr_c","logr_w2","a",\
                "logr_c3","logr_w3","b"],\
        "exponentially_tapered_powerlaw2_ring_gaussian":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "logr_c","logr_w2","logr_w3","a","c"],\
        "exponentially_tapered_powerlaw2_ring2_gaussian":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "logr_c","logr_w2","a",\
                "logr_c3","logr_w3","b",\
		"logr_w4","c"],\
        "exponentially_tapered_powerlaw2_1spiral_dphi_asym_ring":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "logr_c","logr_w2","a","phi_c","phi_w",\
                "phi_0","logr_w_sp","gamma_sp","gamma2_sp","b_pitch","f_az","f_r"],\
        "exponentially_tapered_powerlaw2_1spiral_dphi":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "phi_0","logr_w_sp","gamma_sp","gamma2_sp","b_pitch","f_az","f_r"],\
        "asymmetric_exponentially_tapered_powerlaw2_ring":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "logr_c","logr_w2","a",\
                "phi_c","phi_w"],\
        "asymmetric_exponentially_tapered_powerlaw2_sin_ring":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma","gamma2",\
                "logr_c","logr_w2","a"
                "phi_o"],\
        "broken_powerlaw":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma_in",\
                "gamma_out", "delta"],\
        "broken_powerlaw":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma_in",\
                "gamma_out", "delta"],\
        "exponentially_tapered_broken_powerlaw":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma_in",\
                "gamma_out", "delta", "logr_exp", "gamma2"],\
        "exponentially_tapered_broken_powerlaw_asym_ring":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma_in",\
                "gamma_out", "delta", "logr_exp", "gamma2",\
                "logr_c","logr_w2","a","phi_c","phi_w"],\
        "exponentially_tapered_powerlaw_gaussian4th":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma",\
                "logr_c","logr_w2","loga"],\
        "exponentially_tapered_gaussian4th_powerlaw":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma",\
                "logr_c","logr_w2","loga"],\
        "exponentially_tapered_powerlaw_ring":["x0","y0",\
                "logr_w","logr_c","incl","pa","logflux","logf_a","gamma",\
                 "logr_r","phi_c","phi_w"],\
        "exponentially_tapered_powerlaw_gap":["x0","y0",\
                "logr_w","logr_c","incl","pa","logflux","logflux_r","gamma",\
                "logr_w2"],\
        "gaussian":["x0","y0","logr_w","incl","pa","logflux"],\
        "gaussian_4th":["x0","y0","logr_w","logr_c","incl","pa","logflux"],\
        "gaussian_4th_ring":["x0","y0","logr_w","logr_w2","logr_c","r_r","incl",\
                "pa","logflux","logflux2"],\
        "powerlaw_rectangle":["x0","y0","logx_w","logy_w","pa","logflux",\
                "gamma"],\
        "asymmetric_powerlaw_rectangle":["x0","y0","logx_w","logy_w","pa",\
                "logflux","gamma1","gamma2"],\
        "asymmetric_powerlaw_rectangle_vartrunc":["x0","y0","logx_w","logy_w",\
                "pa","logflux","gamma1","gamma2","gammax","gammay"],\
        "flared_asymmetric_powerlaw_rectangle_vartrunc":["x0","y0","logx_w",\
                "logy_w","pa","logflux","gamma1","gamma2","gammax","gammay",\
                "scale"],\
        "flared_asymmetric_powerlaw_rectangle_vartrunc_varsmooth":["x0","y0",\
                "logx_w","logy_w","pa","logflux","gamma1","gamma2","gammax",\
                "gammay","scale","logsmooth"],\
        "gapped_flared_asymmetric_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma1","gamma2",\
                "gammax","gammay","scale","logsmooth","xgap","logwgap",\
                "logdelta_gap"],\
        "symmetric_gapped_flared_asymmetric_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma1","gamma2",\
                "gammax","gammay","scale","logsmooth","xgap","logwgap",\
                "logdelta_gap"],\
        "broken_powerlaw_rectangle":["x0","y0","logx_w","logy_w","pa",\
                "logflux","gamma_in","gamma_out","logx_t","logdelta"],\
        "asymmetric_broken_powerlaw_rectangle":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta"],\
        "asymmetric_broken_powerlaw_rectangle_vartrunc":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay"],\
        "flared_asymmetric_broken_powerlaw_rectangle_vartrunc":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale"],\
        "flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale",\
                "logsmooth"],\
        "gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale",\
                "logsmooth","xgap","logwgap","logdelta_gap"],\
        "subgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale",\
                "logsmooth","xgap","logwgap","logdelta_gap"],\
        "divgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale",\
                "logsmooth","xgap","logwgap","logdelta_gap"],\
        "symmetric_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale",\
                "logsmooth","xgap","logwgap","logdelta_gap"],\
        "symmetric_subgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale",\
                "logsmooth","xgap","logwgap","logdelta_gap"],\
        "symmetric_divgauss_gapped_flared_asymmetric_broken_powerlaw_rectangle_vartrunc_varsmooth":["x0",\
                "y0","logx_w","logy_w","pa","logflux","gamma_in","gamma_out1",\
                "gamma_out2","logx_t","logdelta","gammax","gammay","scale",\
                "logsmooth","xgap","logwgap","logdelta_gap"],\
        }

for name in names:
    names[name] += ["logx_we","logflux_envelope"]

################################################################################
#
# Revision: define a combine_binary_names function, accept two input model_name_1 and model_name 2
#	    for params in model_name_2, add suffix _2
#	    Combine the params of two models into a list, return the list
#
################################################################################
def combine_binary_names(model_name_1, model_name_2):
    names_1 = names[model_name_1]
    names_2 = [f"{param}_2" for param in names[model_name_2]]
    return names_1 + names_2

################################################################################
#
# Revision: Rename labels as labels_1
# 	    Define a new labels_2 dict, with each key and value added suffix _2
# 	    Combine labels_1 and labels_2 as new labels dict
#
################################################################################    
labels_1 = {
        "x0":"$x_0$",\
        "y0":"$y_0$",\
        "logx_w":"$x_w$",\
        "logy_w":"$y_w$",\
        "logr_w":"$r_w$",\
        "logr_w2":"$r_w2$",\
        "logr_w3":"$r_w3$",\
        "logr_w4":"$r_w4$",\
        "logr_exp":"$r_{exp}$",\
	"logr_c":"$r_c$",\
        "logr_c3":"$r_c3$",\
        "logr_r":"$r_r$",\
        "a":"a",\
        "b":"b",\
        "c":"c",\
        "incl":"$i$",\
        "pa":"$p.a.$",\
        "phi_c":"$\phi_c$",\
        "phi_w":"$\phi_w$",\
        "phi_o":"$\phi_o$",\
        "phi_0":"$\phi_0$",\
        "logr_w_sp":"$logr_{w,sp}$",\
        "gamma_sp":"$\gamma_{sp}$",\
        "gamma2_sp":"$\gamma_{2,sp}$",\
        "b_pitch":"$b_{pitch}$",\
        "f_az":"$f_{az}$",\
        "f_r":"$f_r$",\
        "logflux":r"$F_{\nu}$",\
        "logf_a":"$f_a$",\
        "gamma":"$\gamma$",\
        "gamma1":"$\gamma_1$",\
        "gamma2":"$\gamma_2$",\
        "gammax":"$\gamma_x$",\
        "gammay":"$\gamma_y$",\
        "scale":"scale",\
        "logsmooth":"smooth",\
        "gamma_in":"$\gamma_{in}$",\
        "gamma_out":"$\gamma_{out}$",\
        "gamma_out1":"$\gamma_{out,1}$",\
        "gamma_out2":"$\gamma_{out,2}$",\
        "logx_t":"$x_t$",\
        "delta":"$\Delta$",\
        "xgap":"$x_{gap}$",\
        "logwgap":"$w_{gap}$",\
        "logdelta_gap":"$\Delta_{gap}$",\
        "logx_we":"$x_{w,env}$",\
        "logflux_envelope":r"$F_{\nu,env}$",\
        }

labels_2 = {}
for key, value in labels_1.items():
    new_key = key + "_2"
    new_value = value + "_2"
    labels_2[new_key] = new_value
    
labels = {**labels_1, **labels_2}

priors_1 = {
        "x0":[-0.25,-0.22],\
        "y0":[-0.73,-0.66],\
        "logx_w":[0.0,0.5],\
        "logy_w":[-0.5,"logx_w"],\
        "logr_w":[-2.0, 0.0],\
        "logr_w2":[-3.0,-0.5],\
        "logr_w3":[-4.0,0.0],\
        "logr_w4":[-4.0,0.0],\
        "logr_exp":[-2.0,0.0],\
	"logr_r":[-3.0,1.0],\
        "logr_c":[-3.0,0.0],\
        "logr_c3":[-4.0,1.0],\
        "a":[-1.0,3.0],\
        "b":[-1.0,0.0],\
        "c":[0.0,3.0],\
        "incl":[1.0,numpy.pi/2],\
        "pa":[0.0,0.5],\
        "phi_c":[-numpy.pi,numpy.pi],\
        "phi_w":[0.,numpy.pi],\
        "phi_o":[-numpy.pi,numpy.pi],\
        "phi_0":[-numpy.pi,numpy.pi],\
        "logr_w_sp":[-2.0,2.0],\
        "gamma_sp":[-2.0,2.0],\
        "gamma2_sp":[-2.0,2.0],\
        "b_pitch":[-3.0,3.0],\
        "f_az":[0.0,2.0],\
        "f_r":[0.0,0.5],\
        "logflux":[-3.0,0.0],\
        "logf_a":[-6.,-0.3],\
        "gamma":[-2.0,2.0],\
        "gamma1":[-2.0,2.0],\
        "gamma2":[-12.0,2.0],\
        "gammax":[2.,6.],\
        "gammay":[1.,5.],\
        "scale":[0.,5.],\
        "logsmooth":[-4.,0.],\
        "gamma_in":[0.0,3.0],\
        "gamma_out":[0.0,3.0],\
        "gamma_out1":[-1.,"gamma_in"],\
        "gamma_out2":[-1.,"gamma_in"],\
        "logx_t":[-2.3,"logx_w"],\
        "delta":[0.01,2.0],\
        "xgap":["logx_t","logx_w"],\
        "logwgap":[-3.0,-1.0],\
        "logdelta_gap":[-3.0,0.0],\
        "logx_we":["logr_w",3.0],\
        "logflux_envelope":[-3.,"logflux"],\
        }

priors_2 = {
        "x0_2":[1.05,1.15],\
        "y0_2":[-0.64,-0.48],\
        "logx_w_2":[0.0,0.5],\
        "logy_w_2":[-0.5,"logx_w_2"],\
        "logr_w_2":[-1.7,-0.8],\
        "logr_w2_2":[-3.0,0.0],\
        "logr_w3_2":[-4.0,0.0],\
        "logr_w4_2":[-4.0,0.0],\
	"logr_r_2":[-3.0,1.0],\
        "logr_c_2":[-3.0,1.0],\
        "logr_c3_2":[-4.0,1.0],\
        "a_2":[-2.0,0.0],\
        "b_2":[-2.0,0.0],\
        "c_2":[0.0,3.0],\
        "incl_2":[1.2,1.5],\
        "pa_2":[2.5,numpy.pi],\
        "phi_c_2":[-numpy.pi,numpy.pi],\
        "phi_w_2":[0.,numpy.pi],\
        "phi_o_2":[-numpy.pi,numpy.pi],\
        "logflux_2":[-3.0,-2.6],\
        "logf_a_2":[-6.,-0.3],\
        "gamma_2":[0.0,2.0],\
        "gamma1_2":[-2.0,2.0],\
        "gamma2_2":[-3.0,2.0],\
        "gammax_2":[2.,6.],\
        "gammay_2":[1.,5.],\
        "scale_2":[0.,5.],\
        "logsmooth_2":[-4.,0.],\
        "gamma_in_2":[-2.0,2.],\
        "gamma_out_2":[-2.0,2.],\
        "gamma_out1_2":[-1.,"gamma_in_2"],\
        "gamma_out2_2":[-1.,"gamma_in_2"],\
        "logx_t_2":[-2.3,"logx_w_2"],\
        "delta_2":[-2.,2.],\
        "xgap_2":["logx_t_2","logx_w_2"],\
        "logwgap_2":[-3.0,-1.0],\
        "logdelta_gap_2":[-3.0,0.0],\
        "logx_we_2":[1.202,1.204],\
        "logflux_envelope_2":[-2.980,-2.982],\
        }

priors = {**priors_1, **priors_2}

################################################################################
#
# Create a function which returns a model of the data.
#
################################################################################

def model(u, v, p, npix=256, pixelsize=0.01, output="concat", freq=100.e9, \
        model_name="rectangle"):
    # Get a few parameters that are needed across all models.

    params = dict(zip(names[model_name], p))

    for param in names[model_name]:
        if "log" in param:
            params[param[3:]] = 10.**params[param]

    if not "smooth" in params:
        params["smooth"] = 0.1
    params["smooth"] *= pixelsize

    # Set up the x and y coordinates.

    x = numpy.linspace(-npix/2, npix/2-1, npix) * pixelsize
    y = numpy.linspace(-npix/2, npix/2-1, npix) * pixelsize

    xx, yy = numpy.meshgrid(x, y)

    # Get the coordinates in the frame of the disk.

    xp=xx*numpy.cos(params["pa"])+yy*numpy.sin(params["pa"])
    yp=-xx*numpy.sin(params["pa"])+yy*numpy.cos(params["pa"])

    # Get the geometric model.

    rectangle_component = model_functions[model_name](params, xp, yp)

    if "delta_gap" in params:
        if "subgauss_gap" in model_name:
            if "symmetric" in model_name:
                rectangle_component *= (1 - (1. - params["delta_gap"]) * \
                        numpy.exp(-(numpy.abs(xp) - params["xgap"])**2 / \
                        (2 * params["wgap"]**2)))
            else:
                rectangle_component *= (1. - (1. - params["delta_gap"]) * \
                        numpy.exp(-(xp - params["xgap"])**2 / \
                        (2 * params["wgap"]**2)))
        elif "divgauss_gap" in model_name:
            if "symmetric" in model_name:
                rectangle_component /= ((1./params["delta_gap"] - 1) * \
                        numpy.exp(-(numpy.abs(xp) - params["xgap"])**2 / \
                        (2 * (0.25*params["wgap"])**2)) + 1)
            else:
                rectangle_component /= ((1./params["delta_gap"] - 1) * \
                        numpy.exp(-(xp - params["xgap"])**2 / \
                        (2 * (0.25*params["wgap"])**2)) + 1)
        else:
            if "symmetric" in model_name:
                gap = numpy.where(numpy.logical_and(numpy.abs(xp) > \
                        params["xgap"] - params["wgap"]/2, numpy.abs(xp) < \
                        params["xgap"] + params["wgap"]/2), True, False)
            else:
                gap = numpy.where(numpy.logical_and(xp > params["xgap"] - \
                        params["wgap"]/2, xp < params["xgap"] + \
                        params["wgap"]/2), True, False)

            rectangle_component[gap] *= params["delta_gap"]

    if rectangle_component.sum() > 1.0e-30:
        rectangle_component *= params["flux"] / rectangle_component.sum()

    # Get the large scale Gaussian component.

    V_gauss = uv.model(u, v, [params["x0"], params["y0"], params["x_we"], \
            params["x_we"], 0., params["flux_envelope"]], return_type="data", \
            funct="gauss")

    # Make the intensity image.

    I = im.Image((rectangle_component).reshape(npix,npix,1,1), x=x, y=y, \
            freq=numpy.array([freq]))

    # If we are just making plots of the model, output the image and the 
    # visibilities.

    if pdspy_v2:
        V = uv.interpolate_model(u, v, numpy.array([freq]), I, \
                dRA=params["x0"], dDec=params["y0"])
    else:
        V = uv.interpolate_model(u, v, numpy.array([freq]), I, \
                dRA=-params["x0"], dDec=-params["y0"])

    # Add in the envelope component.

    V.real += V_gauss.real
    V.imag += V_gauss.imag

    # Return the appropriate data.

    if output == "data":
        return V
    elif output == "concat":
        return_val = numpy.concatenate((V.real, V.imag))[:,0]
        del V, V_gauss, I
        gc.collect()
        return return_val
    elif output == 'image':
        return I
        
        
################################################################################
#
# Revision: Define a binary model.
#
# Comments: The input p is from ptform(), which contains the values of parameters, ordered
#           I'll adjust ptform and its input in dynesty sampler,
#	    so that p contains the values of all parameters of two disks, ordered
#	    
#           binary_model() would slice p into two parts, with parameters of each disk in each part
#	    call model() twice for each disk, combine the visibilities of two disks and return
#
################################################################################

def binary_model(u, v, p, npix, pixelsize, output, freq, model_name_1, model_name_2):
    names_1 = names[model_name_1]
    names_2 = [f"{param}_2" for param in names[model_name_2]]
    
    params_value_1 = p[:len(names_1)]
    params_value_2 = p[len(names_1):]
    
    vis_1 = model(u, v, params_value_1, npix, pixelsize, output, freq, model_name_1)
    vis_2 = model(u, v, params_value_2, npix, pixelsize, output, freq, model_name_2)
    
    if output == "data":
        binary_vis = uv.Visibilities(vis_1.u, vis_1.v, vis_1.freq,\
                     vis_1.real + vis_2.real, vis_1.imag + vis_2.imag, vis_1.weights)
    else:
        binary_vis = vis_1 + vis_2

    return binary_vis

################################################################################
#
# Revision: Define a binary likelihood model.
#	    Now accept model_name_1 and model_name_2, and call binary_model()
#
################################################################################

def lnlike(p, x, y, z, zerr, npix, pixelsize, output, freq, model_name_1, model_name_2):
    m = binary_model(x, y, p, npix=npix, pixelsize=pixelsize, output=output, \
            freq=freq, model_name_1=model_name_1, model_name_2=model_name_2)
    kappa=-0.5*(numpy.sum((z - m)**2 * zerr - numpy.log(zerr/(2*numpy.pi))))
    del m
    gc.collect()

    return kappa

################################################################################
#
# Revision: if param in ("x0", "x0_2"):
#
# Comments: For x0, y0, they are defined as 0.0 right now, assume this is true for both disks
#
#	    For params, it is input in dynesty.NestedSampler by:
#		ptform_args=(source, x0, y0, names[model_name])
#
#	    I'll call function combine_binary_names(model_name_1, model_name_2) to replace names[model_name]
#	    So that ptform() accept parameters of two models, where the second model params has suffix _2
#	    
#           ptform would now return an numpy array p
#	    that contains the values of all parameters of two disks, ordered
#
################################################################################

def ptform(u, source, x0, y0, params):
    p = {}
    for i, param in enumerate(params):
        if isinstance(priors[param][1], str):
            if "log" in priors[param][1] and "log" not in param:
                up = 10.**p[priors[param][1]]
            else:
                up = p[priors[param][1]]
        else:
            up = priors[param][1]

        if isinstance(priors[param][0], str):
            if "log" in priors[param][0] and "log" not in param:
                down = 10.**p[priors[param][0]]
            else:
                down = p[priors[param][0]]
        else:
            down = priors[param][0]

        p[param] = (up - down)*u[i] + down

        if param in ("x0", "x0_2"):
            p[param] += x0
        elif param in ("y0", "y0_2"):
            p[param] += y0

    p = numpy.array([p[param] for param in params])
    

    return p


# Define a useful class for plotting.

class Transform:
    def __init__(self, xmin, xmax, dx, fmt):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.fmt = fmt

    def __call__(self, x, p):
        return self.fmt% ((x-(self.xmax-self.xmin)/2)*self.dx)

################################################################################
#
# Set up a pool for parallel runs.
#
################################################################################

withmpi = comm.Get_size() > 1

if withmpi:
    pool = schwimmbad.MPIPool()

    if not pool.is_master():
        pool.wait()
        sys.exit(0)
else:
    pool = None

################################################################################
#
# Parse command line arguments.
#
# Revision: only accept input -m model1&model2, no longer accept single model or "all"
#           directly let model_name_1, model_name_2 = args.model
#
################################################################################

def parse_model_names(model_input):
    model_names = model_input.split('&')
    if len(model_names) == 2:
        return model_names[0], model_names[1]
    else:
        raise argparse.ArgumentTypeError("Invalid model input format. Expected 'model1\&model2'.")

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume', action='store_true')
parser.add_argument('-m', '--model', type=parse_model_names, default=("rectangle", "rectangle"))
parser.add_argument('-s', '--source', type=str, default=sourcename)
parser.add_argument('-f', '--frequency', type=str, default="225GHz")
args = parser.parse_args()

ncpus = comm.Get_size()

sources = args.source.split(",")

model_name_1, model_name_2 = args.model

if args.frequency == "all":
    frequencies = ["15GHz","22GHz","44GHz"]
else:
    frequencies = args.frequency.split(",")

################################################################################
#
# Loop through the sources and fit.
#
# Revision: I already set model_name_1, model_name_2 = args.model
#	    No longer need intertaion for model_name
#
#	    Call combine_binary_names(), store all parameters names into binary_names
#	    
#	    In for loop, replace all names[model_name] with binary_names
#
################################################################################

# Call combine_binary_names(), store all parameters names into binary_names
binary_names = combine_binary_names(model_name_1, model_name_2)

for source, freq in itertools.product(sources, frequencies):
    ############################################################################
    #
    # Read in the data.
    #
    ############################################################################

    # Read in the data.

    vislist = glob.glob(visfilesdir+'*.vis')
    print(vislist)

    invisdata = []
    
    print('Reading in data')
    for visfile in vislist:
       print(visfile)
       invisdata.append(uv.freqcorrect(uv.readvis(visfile)))

    print('Concatentating all the data')
    data = uv.concatenate(invisdata)

    # Average the data to a more manageable size.

#    vis = uv.average(data, gridsize=4096, binsize=4000, mfs=True)
    print('Averaging the vis')
    vis = uv.average(data, gridsize=5000, binsize=4000, mfs=True)

    # Adjust the weights to more accurately represent the correct values.

    print("Weights before are",1./vis.weights.sum()**0.5)

    vis.weights /= weightscale

    print("Weights before are",1./vis.weights.sum()**0.5)
    
    # Read in the image
    print("Reading in fits image for",source)
    image = im.readimfits(imagefile)

    ############################################################################
    #
    # Fit the model to the data.
    #
    ############################################################################

    # Make sure the output directory exists.
    ############################################################################
    #
    # Revision: change the output directory name
    #
    ############################################################################
    output_dir_name = "-".join(args.model)
    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)

    # Set up the inputs for the MCMC function.

    x = vis.u
    y = vis.v
    z = numpy.concatenate((vis.real, vis.imag))[:,0]
    zerr = numpy.concatenate((vis.weights, vis.weights))[:,0]

    N = image.image.shape[0]
    dx = image.header["CDELT2"] * numpy.pi / 180. / arcsec

    # Set up the emcee run.

    print(binary_names)
    ndim, nwalkers = len(binary_names), 400

    # [log10(M_disk), R_in, R_disk, h0, gamma, inclination, position_angle]

#    if freq == "100GHz":
#        x0 = 0.1
#        y0 = -0.1
#    elif source == "L1527old":
#        x0 = 3.75
#        y0 = 3.65
#    else:
    x0 = 0.0
    y0 = 0.0

    # Set up the set of labels for plotting.

    xlabels = [labels[p] for p in binary_names]

    # Set up the Dynesty sampler.

    if args.resume:
        sampler = dynesty.NestedSampler.restore("sampler.save", pool=pool)
        res = sampler.results
    else:
        sampler = dynesty.NestedSampler(lnlike, ptform, ndim, nlive=500, \
            logl_args=(x, y, z, zerr, 256*4, dx/2, "concat", vis.freq[0], \
            model_name_1, model_name_2), ptform_args=(source, x0, y0, binary_names), \
            periodic=[4], pool=pool, sample="rwalk", walks=50)

    # Do the steps in chunks of 5000, and make a plot of the trace as it goes.
    # This way for long models we can see if something is going wrong...

    add_live = False
    while not sampler.added_live:
        sampler.run_nested(dlogz=0.05, checkpoint_file="sampler.save", \
                resume=args.resume, maxiter=1000, add_live=add_live, \
                checkpoint_every=1800)

        # Add the live points and get the results.

        if not add_live:
            sampler.add_final_live()

            res = sampler. results

        # Generate a plot of the trace.

        try:
            fig, ax = dyplot.traceplot(res, show_titles=True, \
                    trace_cmap="viridis", connect=True, \
                    connect_highlight=range(5), labels=xlabels)
        except:
            # If it hasn't converged enough...
            fig, ax = dyplot.traceplot(res, show_titles=True, \
                    trace_cmap="viridis", connect=True, \
                    connect_highlight=range(5), labels=xlabels,\
                    kde=False)

        fig.savefig(output_dir_name+"/{0:s}_{1:s}_traceplot.png".format(source, \
                freq))

        plt.close(fig)

        # Generate a bounds cornerplot.

        fig, ax = dyplot.cornerbound(res, it=res.niter-1, periodic=[5], \
                prior_transform=sampler.prior_transform, show_live=True, \
                labels=xlabels)

        fig.savefig(output_dir_name+"/{0:s}_{1:s}_boundplot.png".format(source, \
                freq))

        plt.close(fig)

        # If we haven't reached the stopping criteria yet, remove the live 
        # points.

        if not add_live:
            sampler._remove_live_points()

        # Manually calculate the stopping criterion to determine whether to
        # add live points.

        logz_remain = numpy.max(sampler.live_logl) + \
                sampler.saved_run["logvol"][-1]
        delta_logz = numpy.logaddexp(sampler.saved_run["logz"][-1], \
                logz_remain) - sampler.saved_run["logz"][-1]

        add_live = delta_logz < 0.05

        # Once we've made it this far, we always want to resume.

        args.resume = True

    # Generate a plot of the weighted samples.

    fig, ax = plt.subplots(ndim-1, ndim-1, figsize=(10,10))

    dyplot.cornerpoints(res, cmap="plasma", kde=False, fig=(fig,ax), \
            labels=xlabels)

    fig.savefig(output_dir_name+"/{0:s}_{1:s}_cornerpoints.png".format(source, freq))

    # Generate a corner plot from Dynesty.

    fig, ax = plt.subplots(ndim, ndim, figsize=(15,15))

    dyplot.cornerplot(res, color="blue", show_titles=True, max_n_ticks=3, \
            quantiles=None, fig=(fig, ax), labels=xlabels)

    fig.savefig(output_dir_name+"/{0:s}_{1:s}_cornerplot.png".format(source, freq))

    # Convert the results to a more traditional set of samples that you would 
    # get from an MCMC program.

    samples, weights = res.samples, numpy.exp(res.logwt - res.logz[-1])

    samples = dyfunc.resample_equal(samples, weights)

    # Save pos, prob, chain.

    numpy.save(output_dir_name+"/{0:s}_{1:s}_samples.npy".format(source, freq), \
            samples)

    numpy.savez(output_dir_name+"/{0:s}_{1:s}_logz.npz".format(source, freq), \
            logz=res["logz"], logzerr=res["logzerr"])

    # Get the best fit parameters and uncertainties.

    params = numpy.median(samples, axis=0)
    sigma = samples.std(axis=0)

    # Write out the results.

    f = open(output_dir_name+"/{0:s}_{1:s}_fit.txt".format(source, freq), "w")
    f.write("Best fit to {0:s} at {1:s}:\n\n".format(source, freq))
    for i, name in enumerate(binary_names):
        f.write("{0:s} = {1:f} +/- {2:f}\n".format(name, params[i], sigma[i]))
    f.write("\nlogz = {0:f} +/- {1:f}\n\n".format(res["logz"][-1], \
            res["logzerr"][-1]))
    f.close()

    print()
    os.system("cat {0:s}/{1:s}_{2:s}_fit.txt".format(output_dir_name, source, freq))

    # Plot histograms of the resulting parameters.

    fig = corner.corner(samples, labels=xlabels, truths=params)

    plt.savefig(output_dir_name+"/{0:s}_{1:s}_fit.pdf".format(source, freq))

    ############################################################################
    #
    # Plot the results.
    #
    ############################################################################

    # Plot the best fit model over the data.

    fig = plt.figure(figsize=(10.35,2.75))

    gs1 = gridspec.GridSpec(1, 1, figure=fig, left=0.07, right=0.28, \
            bottom=0.16, top=0.96)
    gs2 = gridspec.GridSpec(1, 3, figure=fig, left=0.35, right=0.99, \
            bottom=0.16, top=0.96, wspace=0.)

    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs2[0])
    ax3 = plt.subplot(gs2[1])
    ax4 = plt.subplot(gs2[2])

    ax = [ax1, ax2, ax3, ax4]


    data = uv.freqcorrect(data)

    ############################################################################
    #
    # Revision: Now params contains all binary disks parameter best fits
    #           We can call binary_model() to get the overall visibility
    #
    ############################################################################

    # Set the major axis along the x axis.
    import copy
    print("Setting PA = 90")
    params_native = copy.deepcopy(params)
    params_native[0] = 0.0
    params_native[1] = 0.0
    params_native[4] = numpy.pi/2
   
    # Print native model image for the main source.
    model_main_image_native = model(x, y, params[:len(names[model_name_1])], output="image", freq=vis.freq[0], npix=256*4, pixelsize=dx/2, \
            model_name=model_name_1)
    model_main_image_native.header = image.header
    model_main_image_native.asFITS().writeto("model_main_native.fits",overwrite=True)

    model_vis = binary_model(x, y, params, output="data", freq=vis.freq[0], npix=256*4, pixelsize=dx/2, \
            model_name_1=model_name_1, model_name_2=model_name_2)

    # Make a model image.

    model_vis.weights = vis.weights

# cleaning down to some value in the model image
    model_image = uv.clean(model_vis, imsize=N, pixel_size=dx, \
            mode="continuum", mfs=True, convolution="expsinc", \
            weighting="robust", robust=0.0, maxiter=100, \
            threshold=0.000182)[0]
    
    if not pdspy_v2:
        model_image.image = model_image.image[::-1,::-1,:,:]
  
    # write out model to a fits file 
    model_image.header = image.header
    print('Writing model to fits file')
    model_image.asFITS().writeto("model.fits")

    # Make a residual image.
    residuals = uv.Visibilities(vis.u, vis.v, vis.freq, \
            vis.real - model_vis.real, vis.imag - model_vis.imag, \
            vis.weights)
    
    residual_image = uv.clean(residuals, imsize=N, pixel_size=dx, \
                mode="continuum", mfs=True, convolution="expsinc", \
                weighting="robust", robust=0.0, maxiter=100, \
                threshold=0.000100)[0]
    
    if not pdspy_v2:
        residual_image.image = residual_image.image[::-1,::-1,:,:]

    # write out residuals to a fits file so it can be examined in more detail
    residual_image.header = image.header
    print('Writing residuals to fits file')
    residual_image.asFITS().writeto("residuals.fits")

    # Center the data and average the visibilities radially.

    data = uv.center(data, [params[0], params[1], 1.])

    data_1d = uv.average(data, gridsize=40, binsize=500000., radial=True, \
            log=True, logmin=data.uvdist[data.uvdist > 0].min()*0.95, \
            logmax=data.uvdist[data.uvdist > 0].max()*1.05)

    model_vis = uv.center(model_vis, [params[0], params[1], 1.])

    m1d = uv.average(model_vis, gridsize=40, binsize=500000., radial=True, \
            log=True, logmin=data.uvdist[data.uvdist > 0].min()*0.95, \
            logmax=data.uvdist[data.uvdist > 0].max()*1.05)

    # Plot the visibilities.
    print('start to plot visibilities')
    ax[0].errorbar(data_1d.uvdist/1000, data_1d.amp[:,0]*1000, \
            yerr=numpy.sqrt(1./data_1d.weights[:,0])*1000,\
            fmt="k.", markersize=8, markeredgecolor="k")

    # Plot the best fit model
    print('start to plot best fit model')
    ax[0].plot(m1d.uvdist/1000, m1d.amp*1000, "g-")

    # Plot the image.
    
    ticks = numpy.array([-0.7,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.7])

# changed this so the image is not offset
    print('start to plot data image')
    image_min, image_max = numpy.nanmin(image.image), numpy.nanmax(image.image)
    xmin, xmax = int(N/2 + ticks[0]/dx), int(N/2 + ticks[-1]/dx)
    ymin, ymax = int(N/2 + ticks[0]/dx), int(N/2 + ticks[-1]/dx)
    print('image shape = ' + str(image.image.shape))
    ax[1].imshow(image.image[ymin:ymax,xmin:xmax,0,0], origin="lower", \
                    interpolation="none", vmin=image_min, vmax=image_max, cmap="jet")
    print('data range = ' + str(xmin) +', '+ str(xmax) +', '+ str(ymin) +', '+ str(ymax))
    xmin, xmax = int(N/2 + ticks[0]/dx - params[0]/dx), int(N/2 + ticks[-1]/dx \
            - params[0]/dx)
    ymin, ymax = int(N/2 + ticks[0]/dx + params[1]/dx), int(N/2 + ticks[-1]/dx \
            + params[1]/dx)

    print('model, residuals range = ' + str(xmin) +', '+ str(xmax) +', '+ str(ymin) +', '+ str(ymax))
    print('start to plot model image')

    ax[2].imshow(model_image.image[ymin:ymax,xmin:xmax,0,0], origin="lower", \
                    interpolation="none", vmin=image_min, vmax=image_max, cmap="jet")
    
    print('start to plot residuals image')
    ax[3].imshow(residual_image.image[ymin:ymax,xmin:xmax,0,0], origin="lower",\
                    interpolation="none", vmin=image_min, vmax=image_max, cmap="jet")

    transformx = Transform(xmin, xmax, -dx, '%.1f"')
    transformy = Transform(xmin, xmax, dx, '%.1f"')
    print('set ticks')
    for i in [1,2,3]:
        ax[i].set_xticks(ticks[1:-1]/dx + (xmax-xmin-1)/2)
        ax[i].set_yticks(ticks[1:-1]/dx + (xmax-xmin-1)/2)
        ax[i].get_xaxis().set_major_formatter(transformx)
        ax[i].get_yaxis().set_major_formatter(transformy)

    # Label the plots data, model, residual.

    for i, label in zip([1,2,3],["Data","Model","Residual"]):
        ax[i].annotate(label, xy=(0.05,0.9), xycoords="axes fraction", \
                fontsize="large", color="white")

    # Adjust the plot and save it.

    ax[0].axis([50,50000,0,data_1d.amp.max()*1.1*1000])

    ax[0].set_xscale("log", nonpositive='clip')

    ax[0].set_xlabel("U-V Distance [k$\lambda$]", fontsize="large")
    ax[0].set_ylabel("Amplitude [mJy]", fontsize="large")

    for i in range(4):
        ax[i].tick_params(axis='both', which='major', labelsize='large')

    for i in [1,2,3]:
        ax[i].set_xlabel("$\Delta$RA", fontsize="large")
    ax[1].set_ylabel("$\Delta$Dec", fontsize="large")

    for i in [2,3]:
        ax[i].axes.yaxis.set_ticklabels([])

    # Adjust the figure and save.

    fig.savefig(output_dir_name+"/{0:s}_{1:s}_model.pdf".format(source, freq))

# Now we can close the pool.

if withmpi:
    pool.close()
