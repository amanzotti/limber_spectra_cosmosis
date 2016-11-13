'''

Compute DES kappa



'''


from cosmosis.datablock import names, option_section
import numpy as np
import kappa_gals_kernel as lens_kernel
import kappa_cmb_kernel as lens_kernel2
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import hall_CIB_kernel as cib_hall
import sys
import lensing
import util

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.

cosmo = names.cosmological_parameters
distances = names.distances


def cl_limber_x(z_chi, p_kz, l, k1, k2=None, xmin=0.0, xmax=13000.):
    """ calculate the cross-spectrum at multipole l between kernels k1 and k2 in the limber approximation. Comoving distance version. See  cl_limber_z for the redshift version.



        Notes: Here everything is assumed in h units. Maybe not the best choice but that is it.

        Args:
          z_chi: z(chi) redshift as a function of comoving distance.
          hspline: H(z). not used here kept to uniform to cl_limber_z
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

    return scipy.integrate.quad(integrand, xmin, xmax, , limit=100,epsrel=1.49e-05)[0]


def cl_limber_z(chi_z, hspline, rbs, l, k1, k2=None, zmin=0.0, zmax=1100.):
    """ calculate the cross-spectrum at multipole l between kernels k1 and k2 in the limber approximation. redshift  version. See  cl_limber_x for the comoving distance version
   Notes: Here everything is assumed in h units. Maybe not the best choice but that is it.

    Args:
      z_chi: z(chi) redshift as a function of comoving distance.
      hspline: H(z). not used here kept to uniform to cl_limber_z
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

    return scipy.integrate.quad(integrand, zmin, zmax,, limit=100,epsrel=1.49e-05)[0]


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
        option_section, "dndz_filename", default="/Users/alessandromanzotti/Work/Software/cosmosis_new/cosmosis-standard-library/structure/PS_limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt")

    nu = options[option_section, "nu"]
    # nu = np.fromstring(nu, dtype=float, sep=',')
    zc = options.get_double(option_section, "zc", default=2.)
    zs = options.get_double(option_section, "zs", default=2.)
    b = options.get_double(option_section, "b", default=1.)

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
    return (lbins, blockname, zmin, zmax, dndz_filename, nu, zc, zs, b)


def execute(block, config):
    # Just a simple rename for clarity.
    lbins, blockname, zmin, zmax, dndz_filename, nu, zc, zs, b = config
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

    # LOAD DNDZ
    # =======================
    # print dndz_filename
    dndz = np.loadtxt(dndz_filename)
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1])
    norm = scipy.integrate.quad(dndzfun, dndz[0, 0], dndz[-1, 0],limit=100, epsrel=1.49e-03)[0]
    # print norm, dndz[-1, 0]
    # normalize
    dndzfun = interp1d(dndz[:, 0], dndz[:, 1] / norm)

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

    lkern = lens_kernel.kern(dndz[:, 0], dndzfun, chispline, omega_m, h0)
    lkern2 = lens_kernel2.kern(zdist,  omega_m, h0, xlss)
    j2k = 1.e-6 / np.sqrt(83135.)  # for 353
    cib = cib_hall.ssed_kern(
        h0, zdist, chispline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})

    # clCMBDES = np.zeros(np.size(lbins))
    # clCMB = np.zeros(np.size(lbins))
    # clDES = np.zeros(np.size(lbins))

    # # Compute Cl implicit loops on ell
    # # =======================
    # # compute kappa
    # for i, l in util.enumerate_progress(lbins):
    #     clCMBDES[i] = cl_limber_z(chispline, hspline, rbs, l, k1=lkern, k2=lkern2, zmin=dndz[0, 0], zmax=dndz[-1, 0])

    # for i, l in util.enumerate_progress(lbins):
    #     clDES[i] = cl_limber_z(chispline, hspline, rbs, l, k1=lkern, k2=lkern, zmin=dndz[0, 0], zmax=dndz[-1, 0])

    # for i, l in util.enumerate_progress(lbins):
    #     clCMB[i] = cl_limber_z(chispline, hspline, rbs, l, k1=lkern2, k2=lkern2, zmin=dndz[0, 0], zmax=dndz[-1, 0])

    # print clCMBDES
    # print 'cmbxDES done'
    # clDES = [
    #     cl_limber_z(chispline, hspline, rbs, l, k1=lkern, k2=lkern, zmin=dndz[0, 0], zmax=dndz[-1, 0]) for l in lbins]
    # print 'kdes done'
    # clCMB = [
    #     cl_limber_z(chispline, hspline, rbs, l, k1=lkern2, k2=lkern2, zmin=dndz[0, 0], zmax=10.) for l in lbins]
    # print 'kcmb done'



    # =======================
    # SAVE IN DATABLOCK

# copute rho
    # rho =

    # LOAD CMB POWER TO BUILD THE RECOSNTRUCTED B

    ells = block['cmb_cl', 'ell']
    clee = block['cmb_cl', 'ee']
    clpp = block['cmb_cl', 'PP']

    clee *= 2. * np.pi / (ells.astype(float) * (ells.astype(float) + 1.))
    clpp = clpp / (ells.astype(float)) ** 4

    clbb = np.array(lensing.utils.calc_lensed_clbb_first_order(
        lbins, clee, clpp, lmax=ells[-1], nx=1024, dx=2. / 60. / 180. * np.pi).cl, dtype=float)

    obj = 'des_kappa_'
    section = "limber_spectra"
    # block[section, "cl_desk_" + obj] = clDES
    # block[section, "cl_cmbk_" + obj] = clCMB
    # block[section, "cl_desxcmbk_" + obj] = clCMBDES

    # block[section, "ells_" + obj] = lbins
    block[section, "ells_lbins"] = lbins
    block[section, "ells_clbb"] = clbb

    return 0
