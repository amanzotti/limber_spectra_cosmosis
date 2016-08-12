'''

KAPPA galaxies

Part of a series of external utils that creates kernels for Limber integrals. This one is for CMB lensing.

You want to return a spline function W(l,chi,z) with l multipole chi comiving distance z redsfhit which is what is needed for limber.

EVERYTHING IS IN h UNITS

'''

import numpy as np
import scipy
import time
from scipy.interpolate import  interp1d


# def chiint(z, omegam, h0):
#     """
#     Comoving distance integral in case CAMB did not give you the evolution till high redshifts
#     """
#     # 3000 is the c in H/c in h units
#     chiint = 3000. / np.sqrt(omegam * (1. + z) ** 3 + (1. - omegam))
#     return chiint


def wint(z, chiz_in, dndzfun, chispline):
    tmp = dndzfun(z) * (1. - chiz_in / chispline(z))
    return tmp


class kern():

    def __init__(self, zdist, dndzfun, chispline, omm, h0):
        '''
        KAPPA Galaxies KERNEL (h units):

        Args:

            zdist: redshift distribution of the spline
            dndzfun: galaxies redshift distribution
            omm: Omega matter
            h0:hubble constant


        Return:

            kern().w_lxz: kernel for limber integral


       '''

        wb = np.zeros(np.size(zdist))
        # use the z's from the P(k,z) array
        zmax = zdist[np.size(zdist) - 1]
        zmin = zdist[0]
        zmax = zdist[np.size(zdist) - 1]
        self.h0 = h0
        self.omm = omm
        self.zmin = zmin
        self.zmax = zmax
        self.dndzfun = dndzfun
        self.chispline = chispline
        integrate_range = np.linspace(zmin,zmax,int(np.round(zmax-zmin)*80))
        temp_chiz = np.zeros_like(integrate_range)
        for i,z in enumerate(integrate_range):
            temp_chiz[i] = scipy.integrate.quad(wint, z, self.zmax / 1.0001, args=(self.chispline(z), self.dndzfun, self.chispline), limit=200,epsrel=1.49e-05)[0]
        print 'interval',integrate_range[1] - integrate_range[0]
        self.temp_chiz = interp1d(integrate_range, temp_chiz)

    def w_lxz(self, l, x, z):
        '''
        KAPPA gal KERNEL (h units):

        w = 1.5*omegam*h0**2*(1.+z)*chi*\int_z^infty dz' (dn/dz') (1-chi(z)/chi(z'))

       '''
        chiz = self.chispline(z)
        start = time.time()

        # if (z < self.zmax / 1.0001):
        #     tmp = scipy.integrate.quad(
        #         wint, z, self.zmax / 1.0001, args=(chiz, self.dndzfun, self.chispline), limit=200,epsrel=1.49e-05)[0]


        # if (z > self.zmax / 1.0001):
        #     print 'z in kappa_gals_kernel is out of bound'

        # if (z < self.zmin / 1.0001):
        #     print 'less than zmin'

        end = time.time()

        return 1.5 * self.omm * (1. + z) * chiz * self.temp_chiz(z) / (3000. * 3000.)

