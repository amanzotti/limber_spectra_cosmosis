'''

Compute DES kappa - source / desK galaxy

part of the DES projectto do the ratio

STILL PRELIMINARY finish this

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

    # Create DNDZ
    # The idea is to create to dndz for sources and lenses with top hat
    # function source 0.7-1.3 lenses 0.3,0.6 in readshift.

    # for now loading DES and cut above and below
    # =======================
    dndz = np.loadtxt(dndz_filename)
    dndz_lenses = dndz.copy()
    dndz_sources = dndz.copy()
    dndz_sources[(dndz_sources[:, 0] > 1.3) | (dndz_sources[:, 0] < 0.7), 1] = 0.
    dndz_lenses[(dndz_lenses[:, 0] > 0.6) | (dndz_lenses[:, 0] < 0.3), 1] = 0.
    dndzfun= interp1d(dndz[:, 0], dndz[:, 1])

    dndzfun_lenses = interp1d(dndz_lenses[:, 0], dndz_lenses[:, 1])
    dndzfun_sources = interp1d(dndz_sources[:, 0], dndz_sources[:, 1])

    norm = scipy.integrate.quad(dndzfun_sources, dndz_sources[0, 0], dndz_sources[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun_sources = interp1d(dndz_sources[:, 0], dndz_sources[:, 1] / norm)

    norm = scipy.integrate.quad(dndzfun_lenses, dndz_lenses[0, 0], dndz_lenses[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun_lenses = interp1d(dndz_lenses[:, 0], dndz_lenses[:, 1] / norm)

    norm = scipy.integrate.quad(dndzfun, dndz[0, 0], dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1] / norm)





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
    gal_kappa = kappa_gals_kernel.kern(dndz[:, 0], dndzfun_sources, chispline, omega_m, h0)
    lkern = kappa_kernel.kern(zdist, omega_m, h0, xlss)
    # DES bias taken from Giannantonio et
    des_sources = gals_kernel.kern(dndz[:, 0], dndzfun_sources, hspline, omega_m, h0, b=1.0)

    des_lenses = gals_kernel.kern(dndz[:, 0], dndzfun_lenses, hspline, omega_m, h0, b=1.0)

    clkappag = np.zeros(np.size(lbins))
    clkappag_gal = np.zeros(np.size(lbins))
    clkappag_kappacmb = np.zeros(np.size(lbins))
    clkappacmb_gal = np.zeros(np.size(lbins))
    clg_lenses = np.zeros(np.size(lbins))
    clkappa_cmb = np.zeros(np.size(lbins))

    # Compute Cl implicit loops on ell
    # =======================
    # print 'pre k des'

    # p = multiprocessing.Pool(1)
    # def clkappag_parallel(l,chispline=chispline,hspline=hspline,rbs=rbs,k1=gal_kappa,zmin=dndz[0, 0],zmax=dndz[-1, 0]):
    #     import limber_integrals
    #     return limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1, zmin, zmax)


    # # # # clkappag = [clkappag_parallel(l) for l in lbins]
    # # # # sys.exit()
    # p.map(clkappag_parallel,lbins)
    # Parallel(n_jobs=1)(delayed( clkappag_parallel(l) ) for l in lbins)
    # print 'clkappag',clkappag

    clkappag = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=gal_kappa, zmin=dndz[0, 0], zmax=dndz[-1, 0])  for l in lbins]

    print 'clkappag'

    clkappag_gal = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=gal_kappa, k2=des_lenses, zmin=dndz[0, 0], zmax=dndz[-1, 0])  for l in lbins]

    print 'clkappag_gal'


    clkappag_kappacmb = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=gal_kappa, k2=lkern, zmin=dndz[0, 0], zmax=dndz[-1, 0]) for l in lbins]
    print 'clkappag_kappacmb'

    clkappacmb_gal = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=des_lenses, k2=lkern, zmin=dndz[0, 0], zmax=dndz[-1, 0]) for l in lbins]
    print 'clkappacmb_gal'


    clg_lenses = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=des_lenses, zmin=0.3, zmax=.6) for l in lbins]
    print 'clg_lenses'



    clkappa_cmb = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=lkern, zmin=dndz[0, 0], zmax=14.) for l in lbins]
    print 'clkappa_cmb'


    obj = 'ratio'
    section = "limber_spectra"

    block[section, "clkappag_" + obj] = clkappag
    block[section, "clkappag_gal_" + obj] = clkappag_gal
    block[section, "clkappag_kappacmb_" + obj] = clkappag_kappacmb
    block[section, "clkappacmb_gal_" + obj] = clkappacmb_gal
    block[section, "clg_lenses_" + obj] = clg_lenses
    block[section, "clkappa_cmb_" + obj] = clkappa_cmb
    block[section, "ells_" + obj] = lbins



    # sys.exit()



 #    # =======================
 #    # SAVE IN DATABLOCK
 #    degree_sq = 500
 #    rad_sq = 500 * (np.pi / 180) ** 2
 #    fsky = rad_sq / 4. / np.pi
 #    n_gal = 3207184
 #    nlgg = 1. / (3207184. / rad_sq) * np.ones_like(cldes)  # they mention N=2.1 10^-8 in Fosalba Giann

 # #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
 # # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere 4pi rad or 4pi*(180/pi)**2 deg**2

 # # from Giannantonio Fosalba
 # # galaxy number density of 10 arc min^-2

 #    if noisy:
 #        # From Planck model, chenged a little bit to match Blake levels
 #        print clcib
 #        clcib = np.array(clcib) + 525.
 #        print clcib
 #        cldes = np.array(cldes) + nlgg
 #        print 'doing noisy'

 # #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
 # # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere 4pi rad or 4pi*(180/pi)**2 deg**2

 # # from Giannantonio Fosalba
 # # galaxy number density of 10 arc min^-2



    return 0
