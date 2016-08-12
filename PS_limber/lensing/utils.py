import os, copy, glob
import numpy as np
import scipy.ndimage
import maps
import qest
import spec

'''
TEst that cl_unl has the right format as the one that is used in spt. (l+1) factors etc


'''

# from SPT software example

# nx         = 256
# dx         = 2./60./180.*np.pi


def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      - beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             - maximum multipole.
    """
    ls = np.arange(0, lmax+1)
    return np.exp( -(fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.) )

def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
          * noise_uK_arcmin - map noise level in uK.arcmin
          * fwhm_arcmin     - beam full-width-at-half-maximum (fwhm) in arcmin.
          * lmax            - maximum multipole.
    """
    return (noise_uK_arcmin * np.pi/180./60.)**2 / bl(fwhm_arcmin, lmax)**2

def calc_lensed_clbb_first_order(lbins, clee,clpp,lmax, nx =1024, dx = 2./60./180.*np.pi,  w=None):

    # print np.sqrt(clee),np.sqrt(clpp)
    ret = maps.cfft(nx, dx)
    qeep = qest.qest_blm_EP( np.sqrt(clee), np.sqrt(clpp) )
    qeep.fill_resp( qeep, ret, np.ones(lmax+1), 2.*np.ones(lmax+1) )
    # print ret.fft
    return spec.lcl(lmax, ret, dl=1).get_ml(lbins)
    # return ret.get_ml(lbins, w=w)


def calc_lensed_clbb_first_order_curl(lbins, clee,clpp,lmax, nx =1024,  dx =2./60./180.*np.pi,  w=None):
    ret = maps.cfft(nx, dx)
    qeep = qest.qest_blm_EX( np.sqrt(clee), np.sqrt(clpp) )
    qeep.fill_resp( qeep, ret, np.ones(lmax+1), 2.*np.ones(lmax+1) )
    return ret.get_ml(lbins, w=w)