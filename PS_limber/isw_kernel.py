'''

KAPPA CMB

Part of a series of external utils that creates kernels for Limber integrals. This one is for Late ISW.

You want to return a spline function W(l,chi,z) with l multipole chi comiving distance z redsfhit which is what is needed for limber.

EVERYTHING IS IN h UNITS

'''
import numpy as np
import scipy
import util

# ============================================================


class kern():

    def __init__(self, h0, omm, zdist, dzispline, hspline):
        '''
        KAPPA Galaxies KERNEL (h units):

        Args:

            zdist: redshift distribution of the spline
            hspline: H(z) hubble function as a function of redshift
            omm: Omega matter
            h0: Hubble constant
            dzispline: D(z) growth function spline

        Return:

            kern().w_lxz: kernel for limber integral


       '''

        wa = np.zeros(np.size(zdist))
        zmin = zdist[0]
        zmax = zdist[np.size(zdist) - 1]
        self.h0 = h0
        self.omm = omm
        self.zmin = zmin
        self.zmax = zmax
        self.dzispline = dzispline
        self.hspline = hspline

    def w_lxz(self, l, x, z):
        '''
        ISW KERNEL (h units):

        wisw = 3. * (1. + z) *omegam*h0**2 * 1/k**2 d/dz( D(z) (1+z) )/ (D(Z)(1+z))

        This has to be multiplied by H(z) because our convention is that the kernel has to be expressed as \int dchi. ISW is usually define in \int dz

        The last one denominator is needed cause P(z) is a function of z in CosmoSIS is not P(z=0)

        '''

        # yhe 1/ho**3 is coming as in CIB_kernel hall.py to compensate for the h factor.
        # TODO recheck and be sure or, formulate the code to be h independent (no
        # division by h and check hspline the is maybe alrady divided)
        # return self.hspline(z) * 3. * (1. + z) * self.omm / (3000. ** 2) * (x / (l + 0.5)) ** 2 * \
        #     (self.dzispline.derivative(1)(z)) / self.dzispline(z) / self.h0 ** 3

        return (3. * self.omm) * (100. / 3e5) ** 2 * (x / (l + 0.5) ) ** 2 *  (self.dzispline.derivative(1)(z)) / (self.dzispline(z) / (1. + z)) * (self.dzispline(0.) / (1. + 0.))
        # return (3. * self.omm) * (100. / 3e5) ** 2 * (x / (l + 0.5) ) ** 2 * (self.dzispline.derivative(1)(z)) / (self.dzispline(z) / (1. + z))


# ============================================================
# ============================================================
