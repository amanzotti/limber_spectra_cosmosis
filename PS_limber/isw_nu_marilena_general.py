'''

Compute ISW x galaxie

to actually compute the effect of neutrinos on the power spectrum we need to compute a full derivative of the power spectrum.
For this reason here we do not use the dufferent kernel.

TO DO:

'''


from cosmosis.datablock import names, option_section
import numpy as np
import isw_kernel
import gals_kernel
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import matplotlib.pylab as plt
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
    blockname = options.get_string(option_section, "dndz_filename", default="matter_power_lin")

    # What matter power spectrum to use, linear Halofit etc
    dndz_filename = options.get_string(
        option_section, "matter_power", default="/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt")

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

    # getting spline of power spectrum P(k) P(k)(1+z)^2 and D(z)
    rbs = RectBivariateSpline(kpower, zpower, powerarray)
    P_k_a2 = RectBivariateSpline(kpower, zpower, powerarray * (1. + zpower)**2)
    D_k_1pz = RectBivariateSpline(kpower, zpower, np.sqrt(
        powerarray / np.tile(powerarray[:, 0], (np.shape(zpower)[0], 1)).T) * (1. + zpower))

    # test = powerarray / np.tile(powerarray[:, 0], (450, 1)).T
    # print powerarray[:,2]/powerarray[:,0], test[:,2]

    lbins = np.logspace(0.4, 3, 70)
    # Cosmological parameters
    # =======================
    omega_m = block[cosmo, "omega_m"]  # this is cold dark matter + baryons
    omega_nu = block[cosmo, "omega_nu"]
    omega_m += omega_nu
    h0 = block[cosmo, "h0"]
    # =======================

    # =======================
    # print dzispline.derivative(1)(1) / dzispline(0), D_k_1pz.__call__(0.1, 1, dy=1)

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

    # LOAD DNDZ
    # =======================
    z_redshift = np.linspace(0, 10.9, 600)

    dndz_z = gals_kernel.dndz_mari_isw_nu(z_redshift, 2.0)
    dndz = np.column_stack((z_redshift, dndz_z))
    # plt.plot(z_redshift, dndz_z)
    # plt.savefig('test_z.pdf')
    # dndzfun = interp1d(z_redshift, dndz)
    # dndz = np.loadtxt(dndz_filename)
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1])
    # print np.diff(dndz[:, 0])
    norm = scipy.integrate.quad(dndzfun, dndz[0, 0], dndz[-1, 0], limit=100, epsrel=1.49e-03)[0]
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1] / norm)

    # These have dimensions of Mpc; change to h^{-1} Mpc
    d_m *= h0
    h /= h0
    xlss *= h0
    # now in units of h^{-1} Mpc or the inverse
    chispline = interp1d(zdist, d_m)
    hspline = interp1d(zdist, h)
    # =======================

    cliswisw = np.zeros(np.size(lbins))
    clgg = np.zeros(np.size(lbins))
    cliswg = np.zeros(np.size(lbins))
    # print ikern.w_lxz(10, chispline(1.0), 1.0)**2
    # print omega_m, (15 + 0.5) / chispline(2.8)
    print chispline(3.)
    def isw_integrand(z, l):
        x = chispline(z)
        return 1. / x / x * hspline(z) * (3 * omega_m * (100. / 3e5) ** 2 * (x / (l + 0.5)) ** 2) ** 2 * (D_k_1pz.__call__((l + 0.5) / x, z, dx=0, dy=1))**2 * rbs.__call__((l + 0.5) / x, 0)

    def iswg_integrand(z, l):
        b = 1.
        x = chispline(z)
        # return 1. / x / x * hspline(z) * (3 * omega_m * (100. / 3e5) ** 2 * (x *
        # x) / (l + 0.5)**2) * b * dndzfun(z) * P_k_a2.__call__((l + 0.5) / x, z,
        # dx=0, dy=1) / 2. / (1. + z)
        return 1. / x / x * hspline(z) * (3 * omega_m * (100. / 3e5) ** 2 * (x * x) / (l + 0.5)**2) * b * dndzfun(z) * (D_k_1pz.__call__((l + 0.5) / x, z, dx=0, dy=1)) * (D_k_1pz.__call__((l + 0.5) / x, z, dx=0, dy=0)) / (1. + z) * rbs.__call__((l + 0.5) / x, 0)

        # return 1. / x / x * hspline(z) * (3 * omega_m * (100. / 3e5) ** 2 * (x *
        # x) / (l + 0.5)**2) * b * dndzfun(z) * (D_k_1pz.__call__((l + 0.5) / x,
        # z, dx=0, dy=1)) * D_k_1pz.__call__((l + 0.5) / x, z, dx=0, dy=0) / (1. +
        # z) * rbs.__call__((l + 0.5) / x, 0)

    def gg_integrand(z, l):
        x = chispline(z)
        b = 1.
        return 1. / x / x * hspline(z) * b * b * dndzfun(z)**2 * rbs.__call__((l + 0.5) / x, z)


    print dndz[-1, 0]
    # print scipy.__version__
    # Compute Cl implicit loops on ell
    # =======================
    cliswisw = [scipy.integrate.quad(isw_integrand, 0.5, 10.5, args=(l), limit=200, epsrel=1.49e-08)[0] for l in lbins]

    cliswg = [scipy.integrate.quad(iswg_integrand, dndz[0, 0], dndz[-1, 0], args=(l),
                                   limit=200, epsrel=1.49e-08)[0] for l in lbins]

    clgg = [scipy.integrate.quad(gg_integrand, dndz[0, 0], dndz[-1, 0], args=(l),
                                 limit=200, epsrel=1.49e-08)[0] for l in lbins]
    # print clgg
    # print lbins
    # =======================
    # SAVE IN DATABLOCK

    obj = 'fnu_z_2'
    section = "limber_spectra"
    block[section, "cl_isw_" + obj] = cliswisw
    block[section, "cl_iswg_" + obj] = cliswg
    block[section, "cl_g_" + obj] = clgg
    block[section, "D2_z" + obj] = powerarray / np.tile(powerarray[:, 0], (np.shape(zpower)[0], 1)).T
    block[section, "z" + obj] = zpower
    block[section, "k" + obj] = kpower

    block[section, "ells_" + obj] = lbins

# # NOISE Evaluation

#     degree_sq = 500
#     rad_sq = 500 * (np.pi / 180) ** 2
#     fsky = rad_sq / 4. / np.pi
#     n_gal = 3207184
#     nlgg = 1. / (3207184. / rad_sq) * np.ones_like(clgg)  # they mention N=2.1 10^-8 in Fosalba Giann

#  #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
#  # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere 4pi rad or 4pi*(180/pi)**2 deg**2

#  # from Giannantonio Fosalba
#  # galaxy number density of 10 arc min^-2


# # now Delta(Cgt)^2 = Cgg*Ctt/fsky(2\ell+1) see arxiv 0401166

#     deltacgt_sq = (clgg + nlgg) * (cltt - clte_sub) / fsky / (2. * lbins + 1.)
#     block[section, "cl_delta_noise" + obj] = deltacgt_sq

#     S_to_N = np.sqrt(np.sum(np.array(cliswg) ** 2 / deltacgt_sq))

#     print 'Expect detection S/N =', S_to_N
#     print 'maybe it is possible to use CTE correlation'

# # 300 million galaxies in final 5000 deg^2

    return 0
