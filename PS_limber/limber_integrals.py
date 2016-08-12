'''

Module containing the limber integrals definitions used in the PS limber folder

'''


from cosmosis.datablock import names, option_section
import numpy as np
import scipy.integrate
import sys


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
            return 1. / x /x * k1.w_lxz(l, x, z)**2 * self.p_kz(l / x, z)

    else:

        def integrand(x):
            z = z_chi(x)
            return 1. / x /x * k1.w_lxz(l, x, z) * k2.w_lxz(l, x, z) * self.p_kz(l / x, z)


    return scipy.integrate.quad(integrand, xmin, xmax, limit=300, epsrel=1.49e-05)[0]


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
            return 1. / x/x * hspline(z) * k1.w_lxz(l, x, z)**2 * rbs.ev((l + 0.5) / x, z)

    else:

        def integrand(z):
            x = chi_z(z)
            return 1. / x /x * hspline(z) * k1.w_lxz(l, x, z) * k2.w_lxz(l, x, z) * rbs.ev((l + 0.5) / x, z)

    # print 'here' ,integrand(0.5)

    # sys.exit()

    return scipy.integrate.quad(integrand, zmin, zmax, limit=300, epsrel=1.49e-05)[0]
