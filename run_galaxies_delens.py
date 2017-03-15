'''
'''

import pyximport; pyximport.install()

from cosmosis.datablock import names, option_section
import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
import kappa_gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d,InterpolatedUnivariateSpline
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
        option_section, "dndz_filename", default="/home/manzotti/cosmosis/modules/limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt")

    nu = options[option_section, "nu"]
    # nu = np.fromstring(nu, dtype=float, sep=',')
    zc = options.get_double(option_section, "zc", default=2.)
    zs = options.get_double(option_section, "zs", default=2.)
    b = options.get_double(option_section, "b", default=1.)

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
    return (lbins, blockname, zmin, zmax, dndz_filename, nu, zc, zs, b, noisy)


def execute(block, config):
    from profiling.sampling import SamplingProfiler
    profiler = SamplingProfiler()
    profiler.start()
    # Just a simple rename for clarity.
    lbins, blockname, zmin, zmax, dndz_filename, nu, zc, zs, b, noisy = config

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

    # LOAD DNDZ
    # =======================
    dndz = np.loadtxt(dndz_filename)
    dndzfun = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1], ext=2)
    norm = scipy.integrate.quad(dndzfun, dndz[0, 0], dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1] / norm, ext=2)
    # These have dimensions of Mpc; change to h^{-1} Mpc
    d_m *= h0
    h /= h0
    xlss *= h0
    # now in units of h^{-1} Mpc or the inverse
    chispline = InterpolatedUnivariateSpline(zdist[::-1], d_m[::-1], ext=0)
    # z_chi_spline = interp1d(d_m, zdist)
    hspline = InterpolatedUnivariateSpline(zdist[::-1], h[::-1], ext=0)
    # =======================
    # DEFINE KERNELs
    # CIB
    # j2k = 1.e-6 / np.sqrt(83135.)  # for 353
    lkern = kappa_kernel.kern(zdist, hspline, chispline,omega_m, h0, xlss)
    cib = cib_hall.ssed_kern(
        h0, zdist, chispline, hspline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})

    desi_dndz = np.loadtxt("/home/manzotti/cosmosis/modules/limber/data_input/DESI/DESI_dndz.txt")
    desi_dndz[:, 1] = np.sum(desi_dndz[:, 1:], axis=1)

    dndzfun_desi = interp1d(desi_dndz[:, 0], desi_dndz[:, 1])
    norm = scipy.integrate.quad(
        dndzfun_desi, desi_dndz[0, 0], desi_dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun_desi = InterpolatedUnivariateSpline(desi_dndz[:, 0], desi_dndz[:, 1] / norm)
    desi = gals_kernel.kern(desi_dndz[:, 0], dndzfun_desi, hspline, omega_m, h0, b=1.17)

    # DES bias taken from Giannantonio et
    # DES

    des = gals_kernel.kern(dndz[:, 0], dndzfun, chispline, omega_m, h0, b=1.17)

    # Weak lensing

    # SKA
    z_ska = np.linspace(0.01, 6, 600)
    dndzska10 = gals_kernel.dNdZ_parametric_SKA_10mujk(z_ska)
    dndzska1 = gals_kernel.dNdZ_parametric_SKA_1mujk(z_ska)
    dndzska5 = gals_kernel.dNdZ_parametric_SKA_5mujk(z_ska)
    dndzska01 = gals_kernel.dNdZ_parametric_SKA_01mujk(z_ska)

    # ===
    dndzfun = interp1d(z_ska, dndzska01)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # print(norm)
    # normalize
    dndzska01 = InterpolatedUnivariateSpline(z_ska, dndzska01 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=0.1))
    ska01 = gals_kernel.kern(z_ska, dndzska01, hspline, omega_m, h0, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska1)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # print(norm)

    # normalize
    dndzska1 = InterpolatedUnivariateSpline(z_ska, dndzska1 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=1))
    ska1 = gals_kernel.kern(z_ska, dndzska1, hspline, omega_m, h0, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska5)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # print(norm)

    # normalize
    dndzska5 = InterpolatedUnivariateSpline(z_ska, dndzska5 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=5))
    ska5 = gals_kernel.kern(z_ska, dndzska5, hspline, omega_m, h0, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska10)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # print(norm)

    # normalize
    dndzska10 = InterpolatedUnivariateSpline(z_ska, dndzska10 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=10))
    ska10 = gals_kernel.kern(z_ska, dndzska10, hspline, omega_m, h0, b=1.)

    # LSST
    z_lsst = np.linspace(0.01, 10, 200)
    dndzlsst = gals_kernel.dNdZ_parametric_LSST(z_lsst)
    dndzfun = interp1d(z_lsst, dndzlsst)

    norm = scipy.integrate.quad(dndzfun, 0.01, z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # used the same bias model of euclid. Find something better
    dndzlsst = InterpolatedUnivariateSpline(z_lsst, dndzlsst / norm * 1. * np.sqrt(1. + z_lsst))
    lsst = gals_kernel.kern(z_lsst, dndzlsst, hspline, omega_m, h0, b=1.)

    des_weak = kappa_gals_kernel.kern(z_lsst, dndzlsst, chispline, hspline, omega_m, h0)

    # Euclid
    z_euclid = np.linspace(0.01, 5, 200)
    dndzeuclid = gals_kernel.dNdZ_parametric_Euclid(z_euclid)
    dndzfun = interp1d(z_euclid, dndzeuclid)

    norm = scipy.integrate.quad(dndzfun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
    dndzeuclid = InterpolatedUnivariateSpline(z_euclid, dndzeuclid / norm * 1. * np.sqrt(1. + z_euclid))
    # bias montanari et all for Euclid https://arxiv.org/pdf/1506.01369.pdf
    euclid = gals_kernel.kern(z_euclid, dndzeuclid, hspline, omega_m, h0, b=1.)

    # =======
    # Compute Cl implicit loops on ell
    # =======================

    kernels = [lkern, euclid, des_weak, lsst, ska10, ska01, ska5, ska1, cib, desi, des]
    names = ['k', 'euclid', 'des_weak', 'lsst', 'ska10', 'ska01',
             'ska5', 'ska1' 'cib', 'desi', 'des']

    kernels = [lkern, ska10, ska01, ska5, ska1]
    names = ['k', 'ska10', 'ska01',
             'ska5', 'ska1']
    cls = {}
    for i in np.arange(0, len(kernels)):
        for j in np.arange(i, len(kernels)):
            print(names[i], names[j])
            print(max(kernels[i].zmin, kernels[j].zmin), min(kernels[i].zmax, kernels[j].zmax))
            cls[names[i] + names[j]] = [
                limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=kernels[i], k2=kernels[j], zmin=max(kernels[i].zmin, kernels[j].zmin), zmax=min(kernels[i].zmax, kernels[j].zmax)) for l in lbins]

    if noisy:
        print('Adding noise')
        # From Planck model, chenged a little bit to match Blake levels
        # print clcib
        # cls['cibcib'] = np.array(cls['cibcib']) + 525.
        # from
        cls['ska01ska01'] = np.array(cls['ska01ska01']) + 1. / (183868. * 3282.80635)
        cls['ska1ska1'] = np.array(cls['ska1ska1']) + 1. / (65128. * 3282.80635)
        cls['ska5ska5'] = np.array(cls['ska5ska5']) + 1. / (21235. * 3282.80635)
        cls['ska10ska10'] = np.array(cls['ska10ska10']) + 1. / (11849. * 3282.80635)

        #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
        # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere
        # 4pi rad or 4pi*(180/pi)**2 deg**2

        # from Giannantonio Fosalba
        # galaxy number density of 10 arc min^-2

        degree_sq = 500
        rad_sq = degree_sq * (np.pi / 180)**2
        # fsky = rad_sq / 4. / np.pi
        n_gal = 3207184.
        # they mention N=2.1 10^-8 in Fosalba Giann
        # nlgg = 1. / (n_gal / rad_sq) * np.ones_like(cls['desdes'])

        # cls['desdes'] = np.array(cls['desdes']) + nlgg
        # # ===============================================
        # cls['euclideuclid'] = np.array(cls['euclideuclid']) + (30 / (0.000290888)**2)**(-1)
        # cls['lsstlsst'] = np.array(cls['lsstlsst']) + (26 / (0.000290888)**2)**(-1)

    # SAVE
    obj = '_delens'
    section = "limber_spectra"
    for i in np.arange(0, len(kernels)):
        for j in np.arange(i, len(kernels)):
            print(names[i], names[j])
            block[section, "cl_" + names[i] + names[j] + obj] = cls[names[i] + names[j]]

    block[section, "ells_" + obj] = lbins
    profiler.stop()
    profiler.run_viewer()
    return 0

    # =======================
    # SAVE IN DATABLOCK
