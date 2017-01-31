'''

Galaxies PS

Use it as a template for specific galaxies

Part of a series of external utils that creates kernels for Limber integrals. This one is for Galaxies in general.

You want to return a spline function W(l,chi,z) with l multipole chi comiving distance z redsfhit which is what is needed for limber.

EVERYTHING IS IN h UNITS

'''

import numpy as np
import scipy

# === from https://arxiv.org/pdf/1607.01761v1.pdf ===
lsst_noise = 26  # galaxies arcmin^-2
lsst_shape_noise = 0.26  # galaxies arcmin^-2
lsst_area = 18000  # deg^2

#  GENERIC FORMS


def dndz_mari_isw_nu(z, z_m):
    z0 = z_m / 1.4
    return 3. / 2. * z**2 / z0**3 * np.exp(-(z / z0)**1.5)


def dndz_tophat(z, z_min, z_max):
    if (z > z_min) and (z < z_max):
        return 1.
    else:
        return 0.


def dndz_gaussian(z, z0, z_width):
    return np.exp(-(z - z0)**2 / (2 * z_width**2))


def dNdZ_parametric(z, z0, alpha, beta):
    '''
    usual paramteric form of galaxies distribution
    '''
    temp = (z / z0) ** alpha * np.exp(-(z / z0) ** beta)
    return temp


# LSST
def dNdZ_parametric_LSST(z, z0=0.5, alpha=1.27, beta=1.02):
    '''
    usual paramteric form of galaxies distribution
    '''
    temp = (z / z0) ** alpha * np.exp(-(z / z0) ** beta)
    return temp
# EUCLID


def dNdZ_parametric_Euclid(z, zmean=0.9):
    '''
    from     Montanari https://arxiv.org/pdf/1506.01369.pdf
    '''
    z0 = zmean / 1.412
    temp = z ** 2 * np.exp(-(z / z0) ** 1.5)
    return temp

# SKA


def dNdZ_parametric_SKA(z, c2=2.1757, c3=6.6874):
    '''
    from     Montanari https://arxiv.org/pdf/1506.01369.pdf
    this is  is the number of galaxies per redshift and per steradian
    '''
    temp = z ** c2 * np.exp(-c3 * z)
    return temp

coeffs = {}
coeffs[0.1] = [-0.0019, 0.11, 0.20, 0.76]
coeffs[1] = [-0.0020, 0.13, 0.27, 0.81]
coeffs[5] = [-0.0020, 0.16, 0.37, 0.89]
coeffs[10] = [-0.0019, 0.18, 0.43, 0.94]


def dNdZ_SKA_bias(z, mujk):
    '''
    Model:
    b(z) = b3 + b2z + b1z2 + b0z3
    10muJy 5muJy 1muJy 0.1muJy
    -0.0019 0.18 0.43 0.94
    -0.0020 0.16 0.37 0.89
    -0.0020 0.13 0.27 0.81
    -0.0019 0.11 0.20 0.76
    '''
    return coeffs[mujk][3] + coeffs[mujk][2] * z + coeffs[mujk][1] * z**2 + coeffs[mujk][0] * z**3


# total number of sources The total number of radio sources (Ntot) per deg2 given in Table 1
# 11849 10
# 21235 5
# 65128  1
# 183868 0.1

def dNdZ_parametric_SKA_10mujk(z, p0=0.92, p1=1.04, p2=1.11):
    '''
    from     Blake https://arxiv.org/pdf/1511.04653v2.pdf
    '''
    temp = z ** p0 * np.exp(-p1 * z**p2)
    return temp


def dNdZ_parametric_SKA_5mujk(z, p0=1.01, p1=1.14, p2=1.02):
    '''
    from     Blake https://arxiv.org/pdf/1511.04653v2.pdf
    '''
    temp = z ** p0 * np.exp(-p1 * z**p2)
    return temp


def dNdZ_parametric_SKA_1mujk(z, p0=1.18, p1=1.22, p2=0.92):
    '''
    from     Blake https://arxiv.org/pdf/1511.04653v2.pdf
    '''
    temp = z ** p0 * np.exp(-p1 * z**p2)
    return temp


def dNdZ_parametric_SKA_01mujk(z, p0=1.34, p1=1.91, p2=0.64):
    '''
    from     Blake https://arxiv.org/pdf/1511.04653v2.pdf
    '''
    temp = z ** p0 * np.exp(-p1 * z**p2)
    return temp


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
        print(self.norm)
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
        return self.dndzfun(z) * self.b
