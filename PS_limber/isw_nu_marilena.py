'''

Compute ISW x DES



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
    rbs = RectBivariateSpline(kpower, zpower, powerarray)

    lbins = np.logspace(0.4, 3,70)

    # Cosmological parameters
    # =======================
    omega_m = block[cosmo, "omega_m"]
    h0 = block[cosmo, "h0"]
    # =======================

    # Growth
    # =======================
    d_z = block['growth_parameters', "d_z"]
    f_z = block['growth_parameters', "f_z"]
    z_growth = block['growth_parameters', "z"]
    dzispline = InterpolatedUnivariateSpline(z_growth, d_z * (1.+ z_growth), k=5)
    fzispline = InterpolatedUnivariateSpline(z_growth, f_z, k=5)
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

    # LOAD DNDZ
    # =======================
    z_redshift = np.linspace(0, 5, 100)

    dndz_z = gals_kernel.dndz_mari_isw_nu(z_redshift, 0.1)
    dndz = np.column_stack((z_redshift,dndz_z))
    plt.plot(z_redshift,dndz_z)
    plt.savefig('test_z.pdf')
    # dndzfun = interp1d(z_redshift, dndz)
    # dndz = np.loadtxt(dndz_filename)
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1])
    # print np.diff(dndz[:, 0])

    # These have dimensions of Mpc; change to h^{-1} Mpc
    d_m *= h0
    h /= h0
    xlss *= h0
    # now in units of h^{-1} Mpc or the inverse
    chispline = interp1d(zdist, d_m)
    hspline = interp1d(zdist, h)
    # =======================
    # DEFINE KERNELs

    # ells = block['cmb_cl', 'ell']
    # cltt = block['cmb_cl', 'tt']
    # clte = block['cmb_cl', 'te']
    # clee = block['cmb_cl', 'ee']

    # cltt *= 1e-12 * 2. * np.pi / (ells.astype(float) * (ells.astype(float) + 1.))
    # clte *= 1e-12 * 2. * np.pi / (ells.astype(float) * (ells.astype(float) + 1.))
    # clee *= 1e-12 * 2. * np.pi / (ells.astype(float) * (ells.astype(float) + 1.))

    # cltt_func = interp1d(ells, cltt)
    # clte_sub_func = interp1d(ells, clte ** 2 / clee)
    # clte_sub = np.array(clte_sub_func(lbins))
    # cltt = np.array(cltt_func(lbins))

    ikern = isw_kernel.kern(h0, omega_m, zdist,  dzispline, hspline)
    gkern = gals_kernel.kern(dndz[:, 0], dndzfun, hspline, omega_m, h0)
    # gkern = gals_kernel.kern(z_redshift, dndzfun, hspline, omega_m, h0)
    cliswisw = np.zeros(np.size(lbins))
    clgg = np.zeros(np.size(lbins))
    cliswg = np.zeros(np.size(lbins))
    # print ikern.w_lxz(10, chispline(1.0), 1.0)**2
    print limber_integrals.cl_limber_z(chispline, hspline, rbs, 10., k1=ikern, zmin=1e-3, zmax=10.)

    sys.exit()

    # Compute Cl implicit loops on ell
    # =======================

    cliswisw = [
        limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ikern, zmin= 1e-3, zmax=10.) for l in lbins]

    print 'iswisw done'
    cliswg = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ikern, k2=gkern, zmin=dndz[0, 0], zmax=dndz[-1, 0])
              for l in lbins]
    print 'iswg done'

    clgg = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=gkern, zmin=dndz[0, 0], zmax=dndz[-1, 0])
            for l in lbins]
    print 'gg done'

    # =======================
    # SAVE IN DATABLOCK

    obj = 'isw'
    section = "limber_spectra"
    block[section, "cl_isw_" + obj] = cliswisw
    block[section, "cl_iswg_" + obj] = cliswg
    block[section, "cl_g_" + obj] = clgg

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
