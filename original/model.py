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
visfilesdir = '/path-to-UVDATA/IRAS16253/UVDATA/'
imagefile = '/path-to-fits/IRAS16253/IRAS16253_SBLB_continuum_robust_0.0.image.tt0.fits'
# calculate by finding weight scale by 1./vis.weights.sum()**0.5, then matching that to the image RMS
# image RMS = 17.5 uJy/beam, and weights give
weightscale =  8.0
sourcename = 'IRAS16253'
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

def broken_powerlaw_model(params, xp, yp):
    yi = yp / numpy.cos(params["incl"])
    rp = numpy.sqrt(xp**2 + yi**2)

    return params["flux"] * (((numpy.sqrt(rp**2 + params["smooth"]**2)/params["r_w"])**params["gamma_in"]) * (0.5 * \
            (1. + ((numpy.sqrt(rp**2 \
            + params["smooth"]**2)/params["r_w"])**(1./params["delta"])))**((params["gamma_out"]-params["gamma_in"])*params["delta"])))

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
        "broken_powerlaw":broken_powerlaw_model, \
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
        "broken_powerlaw":["x0","y0",\
                "logr_w","incl","pa","logflux","gamma_in",\
                "gamma_out", "logdelta"],\
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

labels = {
        "x0":"$x_0$",\
        "y0":"$y_0$",\
        "logx_w":"$x_w$",\
        "logy_w":"$y_w$",\
        "logr_w":"$r_w$",\
        "logr_w2":"$r_w2$",\
        "logr_w3":"$r_w3$",\
        "logr_c":"$r_c$",\
        "logr_c3":"$r_c3$",\
        "logr_r":"$r_r$",\
        "a":"a",\
        "b":"b",\
        "incl":"$i$",\
        "pa":"$p.a.$",\
        "phi_c":"$\phi_c$",\
        "phi_w":"$\phi_w$",\
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
        "logdelta":"$\Delta$",\
        "xgap":"$x_{gap}$",\
        "logwgap":"$w_{gap}$",\
        "logdelta_gap":"$\Delta_{gap}$",\
        "logx_we":"$x_{w,env}$",\
        "logflux_envelope":r"$F_{\nu,env}$",\
        }

priors = {
        "x0":[0.20,0.22],\
        "y0":[-0.94,-0.92],\
        "logx_w":[0.0,0.5],\
        "logy_w":[-0.5,"logx_w"],\
        "logr_w":[-2.0,0.0],\
        "logr_w2":[-2.0,0.0],\
        "logr_w3":[-1.4,-1.],\
        "logr_r":[-3.0,1.0],\
        "logr_c":[-1.5,0.0],\
        "logr_c3":[-4.0,0.0],\
        "a":[-1.0,0.0],\
        "b":[0.1,0.5],\
        "incl":[0.9,1.4],\
        "pa":[0.0,0.8],\
        "phi_c":[-numpy.pi,numpy.pi/2],\
        "phi_w":[-numpy.pi/2.,numpy.pi/2.],\
        "logflux":[-2.2,-1.6],\
        "logf_a":[-6.,-0.3],\
        "gamma":[0.0,1.0],\
        "gamma1":[-2.,0.],\
        "gamma2":[-3.0,-1.0],\
        "gammax":[2.,6.],\
        "gammay":[1.,5.],\
        "scale":[0.,5.],\
        "logsmooth":[-4.,0.],\
        "gamma_in":[-2.0,2.],\
        "gamma_out":[-2.0,2.],\
        "gamma_out1":[-1.,"gamma_in"],\
        "gamma_out2":[-1.,"gamma_in"],\
        "logx_t":[-2.3,"logx_w"],\
        "logdelta":[-2.,2.],\
        "xgap":["logx_t","logx_w"],\
        "logwgap":[-3.0,-1.0],\
        "logdelta_gap":[-3.0,0.0],\
        "logx_we":["logr_w",3.0],\
        "logflux_envelope":[-3.,"logflux"],\
        }


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

# Define a likelihood function.

def lnlike(p, x, y, z, zerr, npix, pixelsize, output, freq, model_name):
    m = model(x, y, p, npix=npix, pixelsize=pixelsize, output=output, \
            freq=freq, model_name=model_name)
    kappa=-0.5*(numpy.sum((z - m)**2 * zerr - numpy.log(zerr/(2*numpy.pi))))
    del m
    gc.collect()

    return kappa

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

        if param == "x0":
            p[param] += x0
        elif param == "y0":
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
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume', action='store_true')
parser.add_argument('-m', '--model', type=str, default="rectangle")
parser.add_argument('-s', '--source', type=str, default=sourcename)
parser.add_argument('-f', '--frequency', type=str, default="225GHz")
args = parser.parse_args()

ncpus = comm.Get_size()

sources = args.source.split(",")

if args.model == "all":
    models = list(model_functions.keys())
else:
    # Allow for string matching, i.e. we could run all 'broken_powerlaw' models.
    models_input = args.model.split(",")
    """
    models = []
    for m_in in models_input:
        for m_all in list(model_functions.keys()):
            if m_in in m_all:
                models.append(m_all)
    models = list(numpy.unique(models))
    """
    models = models_input

if args.frequency == "all":
    frequencies = ["15GHz","22GHz","44GHz"]
else:
    frequencies = args.frequency.split(",")

################################################################################
#
# Loop through the sources and fit.
#
################################################################################

for source, model_name, freq in itertools.product(sources, models, frequencies):
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
       invisdata.append(uv.readvis(visfile))

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

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    # Set up the inputs for the MCMC function.

    x = vis.u
    y = vis.v
    z = numpy.concatenate((vis.real, vis.imag))[:,0]
    zerr = numpy.concatenate((vis.weights, vis.weights))[:,0]

    N = image.image.shape[0]
    dx = image.header["CDELT2"] * numpy.pi / 180. / arcsec

    # Set up the emcee run.

    print(names[model_name])
    ndim, nwalkers = len(names[model_name]), 400

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

    xlabels = [labels[p] for p in names[model_name]]

    # Set up the Dynesty sampler.

    if args.resume:
        sampler = dynesty.NestedSampler.restore("sampler.save", pool=pool)
        res = sampler.results
    else:
        sampler = dynesty.NestedSampler(lnlike, ptform, ndim, nlive=500, \
            logl_args=(x, y, z, zerr, 256*4, dx/2, "concat", vis.freq[0], \
            model_name), ptform_args=(source, x0, y0, names[model_name]), \
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

        fig.savefig(model_name+"/{0:s}_{1:s}_traceplot.png".format(source, \
                freq))

        plt.close(fig)

        # Generate a bounds cornerplot.

        fig, ax = dyplot.cornerbound(res, it=res.niter-1, periodic=[5], \
                prior_transform=sampler.prior_transform, show_live=True, \
                labels=xlabels)

        fig.savefig(model_name+"/{0:s}_{1:s}_boundplot.png".format(source, \
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

    fig.savefig(model_name+"/{0:s}_{1:s}_cornerpoints.png".format(source, freq))

    # Generate a corner plot from Dynesty.

    fig, ax = plt.subplots(ndim, ndim, figsize=(15,15))

    dyplot.cornerplot(res, color="blue", show_titles=True, max_n_ticks=3, \
            quantiles=None, fig=(fig, ax), labels=xlabels)

    fig.savefig(model_name+"/{0:s}_{1:s}_cornerplot.png".format(source, freq))

    # Convert the results to a more traditional set of samples that you would 
    # get from an MCMC program.

    samples, weights = res.samples, numpy.exp(res.logwt - res.logz[-1])

    samples = dyfunc.resample_equal(samples, weights)

    # Save pos, prob, chain.

    numpy.save(model_name+"/{0:s}_{1:s}_samples.npy".format(source, freq), \
            samples)

    numpy.savez(model_name+"/{0:s}_{1:s}_logz.npz".format(source, freq), \
            logz=res["logz"], logzerr=res["logzerr"])

    # Get the best fit parameters and uncertainties.

    params = numpy.median(samples, axis=0)
    sigma = samples.std(axis=0)

    # Write out the results.

    f = open(model_name+"/{0:s}_{1:s}_fit.txt".format(source, freq), "w")
    f.write("Best fit to {0:s} at {1:s}:\n\n".format(source, freq))
    for i, name in enumerate(names[model_name]):
        f.write("{0:s} = {1:f} +/- {2:f}\n".format(name, params[i], sigma[i]))
    f.write("\nlogz = {0:f} +/- {1:f}\n\n".format(res["logz"][-1], \
            res["logzerr"][-1]))
    f.close()

    print()
    os.system("cat {0:s}/{1:s}_{2:s}_fit.txt".format(model_name, source, freq))

    # Plot histograms of the resulting parameters.

    fig = corner.corner(samples, labels=xlabels, truths=params)

    plt.savefig(model_name+"/{0:s}_{1:s}_fit.pdf".format(source, freq))

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

    model_vis = model(x, y, params, output="data", npix=256*4, pixelsize=dx/2, \
            model_name=model_name)

    # Make a model image.

    model_vis.weights = vis.weights

# cleaning down to some value in the model image
    model_image = uv.clean(model_vis, imsize=N, pixel_size=dx, \
            mode="continuum", mfs=True, convolution="expsinc", \
            weighting="robust", robust=0.0, maxiter=1000, \
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
                weighting="robust", robust=0.0, maxiter=1000, \
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

    ax[0].errorbar(data_1d.uvdist/1000, data_1d.amp[:,0]*1000, \
            yerr=numpy.sqrt(1./data_1d.weights[:,0])*1000,\
            fmt="k.", markersize=8, markeredgecolor="k")

    # Plot the best fit model

    ax[0].plot(m1d.uvdist/1000, m1d.amp*1000, "g-")

    # Plot the image.
    
    ticks = numpy.array([-0.3,-0.2,0.,0.2,0.3])

# changed this so the image is not offset
    xmin, xmax = int(N/2 + ticks[0]/dx), int(N/2 + ticks[-1]/dx)
    ymin, ymax = int(N/2 + ticks[0]/dx), int(N/2 + ticks[-1]/dx)

    ax[1].imshow(image.image[ymin:ymax,xmin:xmax,0,0], origin="lower", \
                    interpolation="none", vmin=image.image.min(), \
                    vmax=image.image.max(), cmap="jet")

    xmin, xmax = int(N/2 + ticks[0]/dx - params[0]/dx), int(N/2 + ticks[-1]/dx \
            - params[0]/dx)
    ymin, ymax = int(N/2 + ticks[0]/dx + params[1]/dx), int(N/2 + ticks[-1]/dx \
            + params[1]/dx)


    ax[2].imshow(model_image.image[ymin:ymax,xmin:xmax,0,0], origin="lower", \
                    interpolation="none", vmin=image.image.min(), \
                    vmax=image.image.max(), cmap="jet")

    ax[3].imshow(residual_image.image[ymin:ymax,xmin:xmax,0,0], origin="lower",\
                    interpolation="none", vmin=image.image.min(), \
                    vmax=image.image.max(), cmap="jet")

    transformx = Transform(xmin, xmax, -dx, '%.1f"')
    transformy = Transform(xmin, xmax, dx, '%.1f"')

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

    fig.savefig(model_name+"/{0:s}_{1:s}_model.pdf".format(source, freq))

# Now we can close the pool.

if withmpi:
    pool.close()
