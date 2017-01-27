'''
'''


from cosmosis.datablock import names, option_section
import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import limber_integrals
# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.


def cl_limber_x(z_chi, p_kz, l, k1, k2=None, xmin=0.0, xmax=13000.):
    """ calculate the cross-spectrum at multipole l between kernels k1 and k2 in the limber approximation. Comoving distance version. See  limber_integrals.cl_limber_z for the redshift version.
        Notes: Here everything is assumed in h units. Maybe not the best choice but that is it.
        Args:
          z_chi: z(chi) redshift as a function of comoving distance.
          hspline: H(z). not used here kept to uniform to limber_integrals.cl_limber_z
          rbs: Power spectrum spline P(k,z) k and P in h units
          l: angular multipole
          k1: First kernel
          k2: Optional Second kernel otherwise k2=k1
          xmin: Min range of integration, comoving distance
          xmax: Max range of integration, comoving distance
        Returns:
          cl_limber : C_l = \int_chi_min^chi_max d\chi {1/\chi^2} K_A(\chi) K_B(\chi)\times P_\delta(k=l/\chi;z)
    """

    if k2 == None:
        k2 = k1

    def integrand(x):
        z = z_chi(x)
        return 1. / x ** 2 * k1.w_lxz(l, x, z) * k2.w_lxz(l, x, z) * self.p_kz(l / x, z)

    return scipy.integrate.quad(integrand, xmin, xmax, limit=300, epsrel=1.49e-05)[0]


def cl_limber_z(chi_z, hspline, rbs, l, k1, k2=None, zmin=0.0, zmax=1100.):
    """ calculate the cross-spectrum at multipole l between kernels k1 and k2 in the limber approximation. redshift  version. See  cl_limber_x for the comoving distance version
   Notes: Here everything is assumed in h units. Maybe not the best choice but that is it.
    Args:
      z_chi: z(chi) redshift as a function of comoving distance.
      hspline: H(z). not used here kept to uniform to limber_integrals.cl_limber_z
      rbs: Power spectrum spline P(k,z) k and P in h units
      l: angular multipole
      k1: First kernel
      k2: Optional Second kernel otherwise k2=k1
      zmin: Min range of integration, redshift
      zmax: Max range of integration, redshift
    Returns:
      cl_limber : C_l = \int_0^z_s dz {d\chi\over dz} {1/\chi^2} K_A(\chi(z)) K_B(\chi(z)\times P_\delta(k=l/\chi(z);z)
    """

    #  TODO check the H factor.

    if k2 == None:
        k2 = k1

    def integrand(z):
        x = chi_z(z)
        return 1. / x ** 2 / hspline(z) * k1.w_lxz(l, x, z) * k2.w_lxz(l, x, z) * rbs.ev((l + 0.5) / x, z)

    return scipy.integrate.quad(integrand, zmin, zmax, limit=300, epsrel=1.49e-05)[0]


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
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1])
    norm = scipy.integrate.quad(dndzfun, dndz[0, 0], dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1] / norm)

    # print np.diff(dndz[:, 0])

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
    # CIB
    j2k = 1.e-6 / np.sqrt(83135.)  # for 353
    lkern = kappa_kernel.kern(zdist, hspline, omega_m, h0, xlss)
    cib = cib_hall.ssed_kern(
        h0, zdist, chispline, hspline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})

    desi_dndz = np.loadtxt("cosmosis-standard-library/structure/PS_limber/data_input/DESI/DESI_dndz.txt")
    desi_dndz[:, 1] = np.sum(desi_dndz[:, 1:], axis=1)

    dndzfun_desi = interp1d(desi_dndz[:, 0], desi_dndz[:, 1])
    norm = scipy.integrate.quad(dndzfun, desi_dndz[0, 0], desi_dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun_desi = interp1d(desi_dndz[:, 0], desi_dndz[:, 1] / norm)
    desi = gals_kernel.kern(desi_dndz[:, 0], dndzfun_desi, hspline, omega_m, h0, b=1.17)

    # DES bias taken from Giannantonio et
    # DES

    des = gals_kernel.kern(dndz[:, 0], dndzfun, hspline, omega_m, h0, b=1.17)

    # SKA
    z_ska = np.linspace(0, 10, 600)
    dndzska10 = gals_kernel.dNdZ_parametric_SKA_10mujk(z_ska)
    dndzska1 = gals_kernel.dNdZ_parametric_SKA_1mujk(z_ska)
    dndzska5 = gals_kernel.dNdZ_parametric_SKA_5mujk(z_ska)
    dndzska01 = gals_kernel.dNdZ_parametric_SKA_01mujk(z_ska)

    # ===
    dndzfun = interp1d(z_ska, dndzska01)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], 6, limit=100, epsrel=1.49e-03)[0]
    # print(norm)
    # normalize
    dndzska01 = interp1d(z_ska, dndzska01 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=0.1))
    ska01 = gals_kernel.kern(z_ska, dndzska01, hspline, omega_m, h0, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska1)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], 6, limit=100, epsrel=1.49e-03)[0]
    # print(norm)

    # normalize
    dndzska1 = interp1d(z_ska, dndzska1 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=1))
    ska1 = gals_kernel.kern(z_ska, dndzska1, hspline, omega_m, h0, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska5)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], 6, limit=100, epsrel=1.49e-03)[0]
    # print(norm)

    # normalize
    dndzska5 = interp1d(z_ska, dndzska5 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=5))
    ska5 = gals_kernel.kern(z_ska, dndzska5, hspline, omega_m, h0, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska10)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], 6, limit=100, epsrel=1.49e-03)[0]
    # print(norm)

    # normalize
    dndzska10 = interp1d(z_ska, dndzska10 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=10))
    ska10 = gals_kernel.kern(z_ska, dndzska10, hspline, omega_m, h0, b=1.)

    # LSST
    z_lsst = np.linspace(0.0, 10, 200)
    dndzlsst = gals_kernel.dNdZ_parametric_LSST(z_lsst)
    dndzfun = interp1d(z_lsst, dndzlsst)

    norm = scipy.integrate.quad(dndzfun, 0.01, 6, limit=100, epsrel=1.49e-03)[0]
    # used the same bias model of euclid. Find something better
    dndzlsst = interp1d(z_lsst, dndzlsst / norm * 1. * np.sqrt(1. + z_lsst))
    lsst = gals_kernel.kern(z_lsst, dndzlsst, hspline, omega_m, h0, b=1.)

    # Euclid
    z_euclid = np.linspace(0.0, 5, 200)
    dndzeuclid = gals_kernel.dNdZ_parametric_Euclid(z_euclid)
    dndzfun = interp1d(z_euclid, dndzeuclid)

    norm = scipy.integrate.quad(dndzfun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
    dndzeuclid = interp1d(z_euclid, dndzeuclid / norm * 1. * np.sqrt(1. + z_euclid))
    # bias montanari et all for Euclid https://arxiv.org/pdf/1506.01369.pdf
    euclid = gals_kernel.kern(z_euclid, dndzeuclid, hspline, omega_m, h0, b=1.)

    # =======

    clkappa = np.zeros(np.size(lbins))

    # clkdes = np.zeros(np.size(lbins))
    # clkcib = np.zeros(np.size(lbins))
    # clkska01 = np.zeros(np.size(lbins))
    # clkska1 = np.zeros(np.size(lbins))
    # clkska5 = np.zeros(np.size(lbins))
    # clkska010 = np.zeros(np.size(lbins))

    clcib = np.zeros(np.size(lbins))
    cldescib = np.zeros(np.size(lbins))
    cldes = np.zeros(np.size(lbins))

    clska01ska01 = np.zeros(np.size(lbins))
    clska5ska5 = np.zeros(np.size(lbins))
    clska1ska1 = np.zeros(np.size(lbins))
    clska10ska10 = np.zeros(np.size(lbins))

    # clska1ska5 = np.zeros(np.size(lbins))
    # clska1ska10 = np.zeros(np.size(lbins))
    # clska1ska01 = np.zeros(np.size(lbins))

    # clska01ska5 = np.zeros(np.size(lbins))
    # clska01ska10 = np.zeros(np.size(lbins))

    # clska5ska10 = np.zeros(np.size(lbins))

    # Compute Cl implicit loops on ell
    # =======================
    print 'pre k des'

    clkappades = [
        limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=lkern, k2=des, zmin=dndz[0, 0], zmax=dndz[-1, 0]) for l in lbins]

    print 'clkappades'
    clkappacib = [(b / np.sqrt(2.4)) *
                  limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=lkern, k2=cib, zmin=dndz[0, 0], zmax=14.) for l in lbins]
    print 'clkappacib'

    clkappa = [
        limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=lkern, zmin=dndz[0, 0], zmax=14.) for l in lbins]
    print 'clkappa'

    cldes = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=des, zmin=dndz[0, 0], zmax=dndz[-1, 0]) for l in lbins]
    print 'cldes'

    cldescib = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=des, k2=cib, zmin=dndz[0, 0], zmax=dndz[-1, 0]) for l in lbins]
    print 'cldescib'

    clcib = [(b / np.sqrt(2.4))**2 * limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=cib, k2=cib, zmin=dndz[0, 0], zmax=14.) for l in lbins]

    print 'clcib'

# SKA
    clska10ska10 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska10, zmin=0.01, zmax=6.) for l in lbins]

    print 'clska10'

    clska5ska5 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska5, zmin=0.01, zmax=6.) for l in lbins]

    print 'clska1'

    clska1ska1 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska1, zmin=0.01, zmax=6.) for l in lbins]

    print 'clska01'

    clska01ska01 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska01, zmin=0.01, zmax=6.) for l in lbins]

    print 'clska01'


# SKA-K
    clkappaska10 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska10, k2=lkern, zmin=0.01, zmax=6.) for l in lbins]

    print 'clskka10'

    clkappaska5 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska5, k2=lkern, zmin=0.01, zmax=6.) for l in lbins]

    print 'clkska5'

    clkappaska1 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska1, k2=lkern, zmin=0.01, zmax=6.) for l in lbins]

    print 'clskka1'

    clkappaska01 = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=ska01, k2=lkern, zmin=0.01, zmax=6.) for l in lbins]

    print 'clkska01'


# LSST

    clkappalsst = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=lsst, k2=lkern, zmin=0.01, zmax=6.) for l in lbins]

    print 'clklsst'
    cllsstlsst = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=lsst, zmin=0.01, zmax=6.) for l in lbins]
    print 'cllsst'


# LSST

    clkappaeuclid = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=euclid, k2=lkern, zmin=0.01, zmax=4.) for l in lbins]

    print 'clkeuclid'
    cleuclideuclid = [limber_integrals.cl_limber_z(chispline, hspline, rbs, l, k1=euclid, zmin=0.01, zmax=4.) for l in lbins]
    print 'clleuclideuclid'

    # =======================
    # SAVE IN DATABLOCK
    degree_sq = 500
    rad_sq = 500 * (np.pi / 180)**2
    fsky = rad_sq / 4. / np.pi
    n_gal = 3207184
    nlgg = 1. / (3207184. / rad_sq) * np.ones_like(cldes)  # they mention N=2.1 10^-8 in Fosalba Giann

 #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
 # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere 4pi rad or 4pi*(180/pi)**2 deg**2

 # from Giannantonio Fosalba
 # galaxy number density of 10 arc min^-2

    if noisy:
        # From Planck model, chenged a little bit to match Blake levels
        # print clcib
        clcib = np.array(clcib) + 525.
        clska01ska01 = np.array(clska01ska01) + 1. / (183868. * 3282.80635)
        clska1ska1 = np.array(clska1ska1) + 1. / (65128. * 3282.80635)
        clska5ska5 = np.array(clska5ska5) + 1. / (21235. * 3282.80635)
        clska10ska10 = np.array(clska10ska10) + 1. / (11849. * 3282.80635)

        # print clcib
        cldes = np.array(cldes) + nlgg
        cleuclideuclid = np.array(cleuclideuclid) + (30 / (0.000290888)**2)**(-1)
        cllsstlsst = np.array(cllsstlsst) + (26 / (0.000290888)**2)**(-1)

        print('doing noisy', (30 / (0.000290888)**2)**(-1), (26 / (0.000290888)**2)**(-1), nlgg)

 #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
 # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere 4pi rad or 4pi*(180/pi)**2 deg**2

 # from Giannantonio Fosalba
 # galaxy number density of 10 arc min^-2

    # print clcib
    obj = '_delens'
    section = "limber_spectra"
    block[section, "cldesk" + obj] = clkappades
    block[section, "clcibk" + obj] = clkappacib
    block[section, "cldesdes" + obj] = cldes
    block[section, "cldescib" + obj] = cldescib
    block[section, "clcibcib" + obj] = clcib
    block[section, "cllsstlsst" + obj] = cllsstlsst
    block[section, "cllsstk" + obj] = clkappalsst
    block[section, "cleuclideuclid" + obj] = cleuclideuclid
    block[section, "clkappaeuclid" + obj] = clkappaeuclid

    block[section, "clk" + obj] = clkappa

    block[section, "clska01k" + obj] = clkappaska01
    block[section, "clska1k" + obj] = clkappaska1
    block[section, "clska5k" + obj] = clkappaska5
    block[section, "clska10k" + obj] = clkappaska10

    block[section, "clska01ska01" + obj] = clska01ska01
    block[section, "clska1ska1" + obj] = clska1ska1
    block[section, "clska5ska5" + obj] = clska5ska5
    block[section, "clska10ska10" + obj] = clska10ska10

    block[section, "ells_" + obj] = lbins

    return 0
