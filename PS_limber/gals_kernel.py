'''

Galaxies PS

Use it as a template for specific galaxies

Part of a series of external utils that creates kernels for Limber integrals. This one is for Galaxies in general.

You want to return a spline function W(l,chi,z) with l multipole chi comiving distance z redsfhit which is what is needed for limber.

EVERYTHING IS IN h UNITS

'''

import numpy as np
import scipy


def dNdZ_parametric(z, z0, alpha, beta):
    '''
    usual paramteric form of galaxies distribution
    '''
    temp = (z / z0) ** alpha * np.exp(-(z / z0) ** beta)
    return temp


def dndz_gaussian(z, z0, z_width):
    return np.exp(-(z - z0)**2 / (2 * z_width**2))


def dndz_mari_isw_nu(z, z_m):
    z0 = z_m / 1.4
    return 3. / 2. * z**2 / z0**3 * np.exp(-(z / z0)**1.5)


def dndz_tophat(z, z_min, z_max):
    if (z > z_min) and (z < z_max):
        return 1.
    else:
        return 0.


class kern():

    def __init__(self, zdist, dndzfun, hspline, omm, h0, b=1.):
        '''
        Galaxies KERNEL (h units):

        Args:

            zdist: redshift distribution of the spline
            dndzfun: galaxies redshift distribution
            omm: Omega matter
            h0:hubble constant
            b: Galaxies bias


        Return:

            kern().w_lxz: kernel for limber integral


       '''

        # use the z's from the P(k,z) array
        zmax = zdist[np.size(zdist) - 1]
        zmin = zdist[0]
        self.h0 = h0
        self.b = b
        self.omm = omm
        self.zmin = zmin
        self.zmax = zmax
        self.dndzfun = dndzfun
        self.norm = scipy.integrate.quad(dndzfun, self.zmin, self.zmax, limit=100, epsrel=1.49e-03)[0]
        print self.norm
        self.hspline = hspline

    def w_lxz(self, l, x, z):
        '''
        Galaxies KERNEL (h units):

        w = dN/dz * b(z)
        Check the following:

        This has to be multiplied by H(z) because our convention is that the kernel has to be expressed as \int dchi. ISW is usually define in \int dz

        with
       '''
        # print z,self.dndzfun(z),  self.norm  , self.b
        return self.dndzfun(z) / self.norm * self.b
