'''

Compute CMB KAPPA

Sep 19 2015

This is a script made to test the importance of high k non linear power spectrum into the c_phi_phi.

It started after an email discussion with marilena



'''


from cosmosis.datablock import names, option_section
import numpy as np
import kappa_cmb_kernel as kappa_kernel
import kappa_gals_kernel as kappa_gals_kernel
import gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import sys
import multiprocessing
from joblib import Parallel, delayed
import limber_integrals

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.

cosmo = names.cosmological_parameters
distances = names.distances




def setup(options):

    # L BINS
    llmin = options.get_double(option_section, "llmin", default=1.)
    llmax = options.get_double(option_section, "llmax", default=3.)
    dlnl = options.get_double(option_section, "dlnl", default=.1)
    # redshift intervals integrals
    zmin = options.get_double(option_section, "zmin", default=1e-2)
    zmax = options.get_double(option_section, "zmax", default=10.)
    blockname = options.get_string(option_section, "matter_power", default="matter_power_nl")

    # What matter power spectrum to use, linear Halofit etc
    dndz_filename = options.get_string(
        option_section, "dndz_filename", default="cosmosis-standard-library/structure/PS_limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt")

    noisy = options.get_bool(option_section, "noisy_spectra", default=True)

    print 'llmin = ', llmin
    print 'llmax = ', llmax
    print 'dlnl = ', dlnl
    print 'zmin ', zmin
    print 'zmax = ', zmax
    print 'matter_power = ', blockname
    print ' '

    # maybe the suffix for saving the spectra
    # zmax (or take it from CAMB)
    # maybe choose between kappa and others

    lbins = np.arange(llmin, llmax, dlnl)
    lbins = 10. ** lbins
    return (lbins, blockname, zmin, zmax, dndz_filename)


def execute(block, config):
    # Just a simple rename for clarity.
    lbins, blockname, zmin, zmax, dndz_filename = config

    # LOAD POWER in h units
    # =======================

    zpower = block[blockname, "z"]
    kpower = block[blockname, "k_h"]
    powerarray = block[blockname, "p_k"].reshape([np.size(zpower), np.size(kpower)]).T
    powerarray[np.where(kpower<0.1),:] = 1e-20
    rbs = RectBivariateSpline(kpower, zpower, powerarray)

    # Cosmological parameters
    # =======================
    omega_m = block[cosmo, "omega_m"]
    h0 = block[cosmo, "h0"]
    # =======================

    # Distances
    # =======================
    # reverse them so they are going in ascending order
    h = block[distances, "h"]
    tmp = h[::-1]
    h = tmp

    xlss = block[distances, "chistar"]
    zdist = block[distances, "z"]
    tmp = zdist[::-1]
    zdist = tmp

    d_m = block[distances, "d_m"]
    tmp = d_m[::-1]
    d_m = tmp
    # =======================
    print np.amax(d_m)



    # These have dimensions of Mpc; change to h^{-1} Mpc
    d_m *= h0
    h /= h0
    xlss *= h0
    # now in units of h^{-1} Mpc or the inverse
    chispline = interp1d(zdist, d_m)
    z_chi_spline = interp1d(d_m, zdist)
    hspline = interp1d(zdist, h)
    # =======================
    # DEFINE KERNELs
    lkern = kappa_kernel.kern(zdist, omega_m, h0, xlss)


    clkappacmb= np.zeros(np.size(lbins))

    clkappa_cmb = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=lkern, zmin=0.0, zmax=20.) for l in lbins]
    print 'clkappa_cmb'


    obj = 'ratio'
    section = "limber_spectra"

    block[section, "clkappa_cmb_" + obj] = clkappa_cmb
    block[section, "ells_" + obj] = lbins



    # sys.exit()



    return 0

