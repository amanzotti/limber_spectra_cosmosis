'''

LENSING CMB

Part of a series of external utils that creates kernels for Limber integrals. This is for CMB lensing.

You want to return a spline which is what is needed for limber.


'''

import numpy as np
import scipy


def chiint(z, omegam, h0):
    chiint = 3000. / np.sqrt(omegam * (1. + z) ** 3 + (1. - omegam))
    return chiint


class kern():

    def __init__(self, zdist, omm, h0, xlss):
        wb = np.zeros(np.size(zdist))
        # use the z's from the P(k,z) array
        zmax = zdist[np.size(zdist) - 1]
        zmin = zdist[0]
        zmax = zdist[np.size(zdist) - 1]
        self.h0 = h0
        self.omm = omm
        self.zmin = zmin
        self.zmax = zmax
        self.xlss = xlss
        chicmb = xlss

        for i, z in enumerate(zdist):
            z = z + .001
            chiz = scipy.integrate.quad(chiint, 0., z, args=(omm, h0))[0]
            tmp = 0.
            if (z < zmax / 1.0001):
                tmp = (1. - chiz / chicmb)
            wb[i] = 1.5 * omm * (1. + z) * chiz * tmp / 3000. ** 2
        self.w = wb
        self.w_interp = scipy.interpolate.interp1d(zdist, wb)

    def w_lxz(self, l, x, z):
        chiz = scipy.integrate.quad(chiint, 0., z, args=(self.omm, self.h0))[0]
        if (z < self.zmax / 1.0001):
            tmp = (1. - chiz / self.xlss)
        return 3. * self.omm * (1. + z) / chiz * (chiz / l) ** 2 * tmp / 3000. ** 2
