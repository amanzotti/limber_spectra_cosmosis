'''

Compute DES kappa



'''

import pyximport
pyximport.install(reload_support=True)
from cosmosis.datablock import names, option_section
import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
reload(gals_kernel)
import kappa_gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
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
    # # What matter power spectrum to use, linear Halofit etc
    # dndz_filename = options.get_string(
    # option_section, "dndz_filename",
    # default="/Users/alessandromanzotti/Work/Software/cosmosis_new/cosmosis-standard-library/structure/PS_limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt")

    # nu = options[option_section, "nu"]
    # # nu = np.fromstring(nu, dtype=float, sep=',')
    # zc = options.get_double(option_section, "zc", default=2.)
    # zs = options.get_double(option_section, "zs", default=2.)
    # b = options.get_double(option_section, "b", default=1.)

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
    return (lbins, blockname, zmin, zmax)  # , dndz_filename, nu, zc, zs, b)


def execute(block, config):
    # Just a simple rename for clarity.
    lbins, blockname, zmin, zmax = config
    # LOAD POWER in h units
    # =======================
    # LOAD POWER in h units
    # =======================

    zpower = block[blockname, "z"]
    kpower = block[blockname, "k_h"]
    print blockname
    powerarray = block[blockname, "p_k"].reshape([np.size(zpower), np.size(kpower)]).T
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

    d_m *= h0
    h /= h0
    xlss *= h0
    # now in units of h^{-1} Mpc or the inverse
    chispline = InterpolatedUnivariateSpline(zdist[::-1], d_m[::-1], ext=0)
    # z_chi_spline = interp1d(d_m, zdist)
    hspline = InterpolatedUnivariateSpline(zdist[::-1], h[::-1], ext=0)
    # =======================

    # LOAD DNDZ
    # =======================
    # print dndz_filename
    z = np.linspace(0.01, 2)
    z_mean = 1.
    z_sigma = 0.3
    dndzfun = interp1d(z, np.exp(-(z - z_mean)**2 / z_sigma**2))
    norm = scipy.integrate.quad(dndzfun, 0.01, 2, limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun = interp1d(z, np.exp(-(z - z_mean)**2 / z_sigma**2) / norm)
    print scipy.integrate.quad(dndzfun, 0.01, 2, limit=100, epsrel=1.49e-03)[0]

    gals = gals_kernel.kern(z, dndzfun, hspline, omega_m, h0, b=1.)

    # print np.diff(dndz[:, 0])

    # =======================
    # DEFINE KERNELs
    z = np.linspace(0.01, 4)

    z_mean = 0.7
    z_sigma = 0.2
    dndzfun = interp1d(z, np.exp(-(z - z_mean)**2 / z_sigma**2))
    norm = scipy.integrate.quad(dndzfun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
    # print norm, dndz[-1, 0]
    # normalize
    dndzfun = interp1d(z, np.exp(-(z - z_mean)**2 / z_sigma**2) / norm)
    print scipy.integrate.quad(dndzfun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]

    lkern = kappa_gals_kernel.kern(z, dndzfun, chispline, hspline, omega_m, h0)

    # j2k = 1.e-6 / np.sqrt(83135.)  # for 353
    # cib = cib_hall.ssed_kern(
    #     h0, zdist, chispline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})

    # clCMBDES = np.zeros(np.size(lbins))
    # clCMB = np.zeros(np.size(lbins))
    cl_desk = [limber_integrals.cl_limber_z(
        chispline, hspline, rbs, l, k1=lkern, k2=gals, zmin=0.01, zmax=2) for l in lbins]
    cl_kk = [limber_integrals.cl_limber_z(
        chispline, hspline, rbs, l, k1=lkern, zmin=0.01, zmax=2.) for l in lbins]
    cl_desdes = [limber_integrals.cl_limber_z(
        chispline, hspline, rbs, l, k1=gals, zmin=0.01, zmax=2.) for l in lbins]

    # LOAD CMB POWER TO BUILD THE RECOSNTRUCTED B

    # ells = block['cmb_cl', 'ell']
    # clee = block['cmb_cl', 'ee']
    # clpp = block['cmb_cl', 'PP']

    # clee *= 2. * np.pi / (ells.astype(float) * (ells.astype(float) + 1.))
    # clpp = clpp / (ells.astype(float)) ** 4

    # clbb = np.array(lensing.utils.calc_lensed_clbb_first_order(
    #     lbins, clee, clpp, lmax=ells[-1], nx=1024, dx=2. / 60. / 180. * np.pi).cl, dtype=float)

    obj = 'test_kappa'
    section = "limber_spectra"
    block[section, "cl_desk_" + obj] = cl_desk
    block[section, "cl_kk" + obj] = cl_kk
    block[section, "cl_desdes" + obj] = cl_desdes

    # block[section, "ells_" + obj] = lbins
    block[section, "ells_lbins"] = lbins
    # block[section, "ells_clbb"] = clbb

    return 0
