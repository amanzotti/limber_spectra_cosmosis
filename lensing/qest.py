import numpy as np
import maps

# ------------------------------
# Obsolete, delete soon (04/24/2014

# def qe_cov_fill_helper( qe1, qe2, cfft, f1, f2, switch_12=False, switch_34=False, conj_12=False, conj_34=False ):
#     lx, ly = cfft.get_lxly()
#     l      = np.sqrt(lx**2 + ly**2)
#     psi    = np.arctan2(lx, -ly)

#     if f1.shape != l.shape:
#         assert( len(f1.shape) == 1 )
#         f1 = np.interp( l.flatten(), np.arange(0, len(f1)), f1, right=0 ).reshape(l.shape)

#     if f2.shape != l.shape:
#         assert( len(f2.shape) == 1 )
#         f2 = np.interp( l.flatten(), np.arange(0, len(f2)), f2, right=0 ).reshape(l.shape)

#     i0_12 = { False : 0, True : 1 }[switch_12]; i0_34 = { False : 0, True : 1 }[switch_34]
#     i1_12 = { False : 1, True : 0 }[switch_12]; i1_34 = { False : 1, True : 0 }[switch_34]

#     cfunc_12 = { False : lambda v : v, True : lambda v : np.conj(v) }[conj_12]
#     cfunc_34 = { False : lambda v : v, True : lambda v : np.conj(v) }[conj_34]

#     fft = cfft.fft
#     for i in xrange(0, qe1.ntrm):
#         for j in xrange(0, qe2.ntrm):
#             fft[:,:] += np.fft.fft2(
#                 np.fft.ifft2(
#                     cfunc_12(qe1.wl[i][i0_12](l, lx, ly)) *
#                     cfunc_34(qe2.wl[j][i0_34](l, lx, ly)) * f1 *
#                     np.exp(+1.j*((-1)**(conj_12)*qe1.sl[i][i0_12]+(-1)**(conj_34)*qe2.sl[j][i0_34])*psi) ) *
#                 np.fft.ifft2(
#                     cfunc_12(qe1.wl[i][i1_12](l, lx, ly)) *
#                     cfunc_34(qe2.wl[j][i1_34](l, lx, ly)) * f2 *
#                     np.exp(+1.j*((-1)**(conj_12)*qe1.sl[i][i1_12]+(-1)**(conj_34)*qe2.sl[j][i1_34])*psi) )
#                 ) * ( cfunc_12(qe1.wl[i][2](l, lx, ly)) *
#                       cfunc_34(qe2.wl[j][2](l, lx, ly)) *
#                       np.exp(-1.j*((-1)**(conj_12)*qe1.sl[i][2]+(-1)**(conj_34)*qe2.sl[j][2])*psi) ) * 0.25 / (cfft.dx * cfft.dy)

#     return cfft


def pad_ft(a,npad=2):
    '''
    Pad an array in Fourier-space for FFT convolutions
    '''
    if npad==1: return a
    nx,ny = a.shape
    p = np.zeros([nx*npad,ny*npad], dtype=a.dtype)
    p[0:nx,0:ny] = np.fft.fftshift(a)
    p = np.roll(np.roll(p,-nx/2,axis=0),-ny/2,axis=1)
    return p

def unpad_ft(a,npad=2):
    '''
    Un-pad an array in Fourier-space
    '''
    if npad==1: return a
    nx_pad,ny_pad = a.shape
    nx = int(nx_pad/npad); ny=int(ny_pad/npad)
    return np.roll(np.roll(
            (np.roll(np.roll(a,nx/2,axis=0),ny/2,axis=1)[0:nx,0:ny]),
            nx/2,axis=0),ny/2,axis=1)

def iconvolve_padded(f, g,npad=2):
    '''
    Calculate the convolution:
       ret(L) = iint{d^2\vec{l} f(l) \times g(L-l)}
    '''
    return (unpad_ft(np.fft.fft2(
                np.fft.ifft2(pad_ft(f,npad=npad)) *
                np.fft.ifft2(pad_ft(g,npad=npad))),
                     npad=npad)*npad**2)


def qe_cov_fill_helper( qe1, qe2, cfft, f1, f2, switch_12=False, switch_34=False, conj_12=False, conj_34=False, npad=None):
    '''
    Calculate the covariance between two estimators.
    When used to calculate the response, the return value is half of the full response.
    '''
    if npad==None: # return the maximum npad of qe1 or qe2.  If either has no specified npad, default to 2.
        if hasattr(qe1, 'npad_conv'): npad_qe1 = qe1.npad_conv
        else: npad_qe1 = 2
        if hasattr(qe2, 'npad_conv'): npad_qe2 = qe2.npad_conv
        else: npad_qe2 = 2

        npad = max(npad_qe1, npad_qe2)
        if npad < 1: npad=1 # never less than 1

    if npad != 2:
        print "lensing.qest.qe_cov_fill_helper(): npad is not equal to 2!  I hope you know what you are doing..."

    lx, ly = cfft.get_lxly()
    l      = np.sqrt(lx**2 + ly**2)
    psi    = np.arctan2(lx, -ly)
    nx, ny = l.shape

    if f1.shape != l.shape:
        assert( len(f1.shape) == 1 )
        f1 = np.interp( l.flatten(), np.arange(0, len(f1)), f1, right=0 ).reshape(l.shape)

    if f2.shape != l.shape:
        assert( len(f2.shape) == 1 )
        f2 = np.interp( l.flatten(), np.arange(0, len(f2)), f2, right=0 ).reshape(l.shape)

    i0_12 = { False : 0, True : 1 }[switch_12]; i0_34 = { False : 0, True : 1 }[switch_34]
    i1_12 = { False : 1, True : 0 }[switch_12]; i1_34 = { False : 1, True : 0 }[switch_34]

    cfunc_12 = { False : lambda v : v, True : lambda v : np.conj(v) }[conj_12]
    cfunc_34 = { False : lambda v : v, True : lambda v : np.conj(v) }[conj_34]

    fft = cfft.fft

    for i in xrange(0, qe1.ntrm):
        for j in xrange(0, qe2.ntrm):
            term1 = (cfunc_12(qe1.wl[i][i0_12](l, lx, ly)) *
                  cfunc_34(qe2.wl[j][i0_34](l, lx, ly)) * f1 *
                  np.exp(+1.j*((-1)**(conj_12)*qe1.sl[i][i0_12]+(-1)**(conj_34)*qe2.sl[j][i0_34])*psi))
            term2 = (cfunc_12(qe1.wl[i][i1_12](l, lx, ly)) *
                  cfunc_34(qe2.wl[j][i1_34](l, lx, ly)) * f2 *
                  np.exp(+1.j*((-1)**(conj_12)*qe1.sl[i][i1_12]+(-1)**(conj_34)*qe2.sl[j][i1_34])*psi))

            fft[:,:] += (iconvolve_padded(term1, term2, npad=npad)*
                         ( cfunc_12(qe1.wl[i][2](l, lx, ly)) *
                           cfunc_34(qe2.wl[j][2](l, lx, ly)) *
                           np.exp(-1.j*((-1)**(conj_12)*qe1.sl[i][2]+(-1)**(conj_34)*qe2.sl[j][2])*psi) ) * 0.25 / (cfft.dx * cfft.dy))

    return cfft


class qest():
    def __init__(self):
        pass

    def eval( self, r1, r2, npad=None ):
        '''
        Evaluate the quadradic estimator of \phi from fields r1 and r2
        '''
        if hasattr(r1, 'get_cfft'):
            r1 = r1.get_cfft()
        if hasattr(r2, 'get_cfft'):
            r2 = r2.get_cfft()

        assert( r1.compatible(r2) )

        if npad==None:
            if self.npad_conv: npad=self.npad_conv
            else: npad=2

        cfft   = maps.cfft( r1.nx, r1.dx, ny=r1.ny, dy=r1.dy )

        lx, ly = cfft.get_lxly()
        l      = np.sqrt(lx**2 + ly**2)
        psi    = np.arctan2(lx, -ly)

        fft = cfft.fft

        for i in xrange(0, self.ntrm):
            term1 = self.wl[i][0](l, lx, ly) * r1.fft * np.exp(+1.j*self.sl[i][0]*psi)
            term2 = self.wl[i][1](l, lx, ly) * r2.fft * np.exp(+1.j*self.sl[i][1]*psi)

            fft[:,:] += (iconvolve_padded(term1, term2, npad=npad)*
                         ( self.wl[i][2](l, lx, ly) *
                           np.exp(-1.j*self.sl[i][2]*psi) ) * 0.5 / np.sqrt(cfft.dx * cfft.dy) * np.sqrt(cfft.nx * cfft.ny))

        # # Original code ---------------------
        # fft_orig = fft*0.
        # for i in xrange(0, self.ntrm):
        #     fft_orig[:,:] += np.fft.fft2(
        #         np.fft.ifft2(
        #             self.wl[i][0](l, lx, ly) * r1.fft *
        #             np.exp(+1.j*self.sl[i][0]*psi) ) *
        #         np.fft.ifft2(
        #             self.wl[i][1](l, lx, ly) * r2.fft *
        #             np.exp(+1.j*self.sl[i][1]*psi) )
        #         ) * ( self.wl[i][2](l, lx, ly) *
        #               np.exp(-1.j*self.sl[i][2]*psi) ) * 0.5 / np.sqrt(cfft.dx * cfft.dy) * np.sqrt(cfft.nx * cfft.ny)
        # # -----------------------------------

        return cfft

    def fill_resp( self, qe2, cfft, f1, f2, npad=2 ):
        print "fill_resp", qe2
        cfft.fft[:,:] = 0.0
        qe_cov_fill_helper( self, qe2, cfft, f1, f2, npad=npad)
        cfft.fft[:,:] *= 2.0 # Multiply by 2 because qe_cov_fill_helper returns 1/2 the response.
        return cfft

    def fill_clqq( self, cfft, f11, f12, f22, npad=2):
        cfft.fft[:,:] = 0.0
        qe_cov_fill_helper( self, self, cfft, f11, f22, switch_34=False, conj_34=True, npad=npad )
        qe_cov_fill_helper( self, self, cfft, f12, f12, switch_34=True,  conj_34=True, npad=npad )
        return cfft

    def get_sl1(self, i):
        return self.sl[i][0]

    def get_sl2(self, i):
        return self.sl[i][1]

    def get_slL(self, i):
        return self.sl[i][2]

    def get_wl1(self, i, l, lx, ly):
        return self.wl[i][0](l, lx, ly)

    def get_wl2(self, i, l, lx, ly):
        return self.wl[i][1](l, lx, ly)

    def get_wlL(self, i, l, lx, ly):
        return self.wl[i][2](l, lx, ly)


#############################
# Classes for calculating quadratic estimates.
# In order to write the QE convolutions as FFT's we separate the equations into terms,
#   where each term can be expressed as components of the form
#        F(l) * G(L-l) * H(L)
#   Here, l is the integration variable.
# components self.wl[:][0] correspond to F(l)
# components self.wl[:][1] correspond to G(L-l)
# components self.wl[:][2] are independend of l, corresponding to H(L)
#
# self.npad_conv is the factor for padding the convolution-by-FFT calculations
#############################

class qest_plm_TT_s0(qest):
    def __init__(self, cltt):
        self.cltt = cltt
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : [0,0,0] for i in xrange(0,self.ntrm) }

        self.wl[0][0] = self.wc_lx
        self.wl[0][1] = self.wo_m1
        self.wl[0][2] = self.wo_lx

        self.wl[1][0] = self.wc_ly
        self.wl[1][1] = self.wo_m1
        self.wl[1][2] = self.wo_ly

        self.wl[2][0] = self.wo_m1
        self.wl[2][1] = self.wc_lx
        self.wl[2][2] = self.wo_lx

        self.wl[3][0] = self.wo_m1
        self.wl[3][1] = self.wc_ly
        self.wl[3][2] = self.wo_ly

        self.npad_conv = 2

    def wo_m1(self, l, lx, ly):
        return 1.0
    def wo_lx(self, l, lx, ly):
        return lx
    def wc_lx(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * lx
    def wo_ly(self, l, lx, ly):
        return ly
    def wc_ly(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * ly

class qest_plm_TT(qest):
    def __init__(self, cltt):
        self.cltt = cltt
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +1
        self.wl[0][1] = self.wo_d2; self.sl[0][1] = +0
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -1
        self.wl[1][1] = self.wo_d2; self.sl[1][1] = +0
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wo_d2; self.sl[2][0] = +0
        self.wl[2][1] = self.wc_ml; self.sl[2][1] = +1
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wo_d2; self.sl[3][0] = +0
        self.wl[3][1] = self.wc_ml; self.sl[3][1] = -1
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2

    def wo_d2(self, l, lx, ly):
        return -0.5
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * l

class qest_xlm_TT(qest):
    def __init__(self, cltt):
        self.cltt = cltt
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +1
        self.wl[0][1] = self.wo_d2; self.sl[0][1] = +0
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -1
        self.wl[1][1] = self.wo_n2; self.sl[1][1] = +0
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wo_d2; self.sl[2][0] = +0
        self.wl[2][1] = self.wc_ml; self.sl[2][1] = +1
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wo_n2; self.sl[3][0] = +0
        self.wl[3][1] = self.wc_ml; self.sl[3][1] = -1
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

    def wo_d2(self, l, lx, ly):
        return -0.5j
    def wo_n2(self, l, lx, ly):
        return +0.5j
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * l

class qest_plm_TE(qest):
    def __init__(self, clte):
        self.clte = clte
        self.ntrm = 6

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_d4; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_d4; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_d4; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_d4; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        # dt e
        self.wl[4][0] = self.wo_d2; self.sl[4][0] = +0
        self.wl[4][1] = self.wc_ml; self.sl[4][1] = +1
        self.wl[4][2] = self.wo_ml; self.sl[4][2] = +1

        self.wl[5][0] = self.wo_d2; self.sl[5][0] = +0
        self.wl[5][1] = self.wc_ml; self.sl[5][1] = -1
        self.wl[5][2] = self.wo_ml; self.sl[5][2] = -1

        self.npad_conv = 2

    def wo_d2(self, l, lx, ly):
        return -0.50
    def wo_d4(self, l, lx, ly):
        return -0.25
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * l

class qest_plm_ET(qest_plm_TE):
    def __init__(self, clte):
        self.clte = clte
        self.ntrm = 6

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][0] = self.wo_d4; self.sl[0][0] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][0] = self.wo_d4; self.sl[1][0] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][1] = self.wc_ml; self.sl[2][1] = -1
        self.wl[2][0] = self.wo_d4; self.sl[2][0] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][1] = self.wc_ml; self.sl[3][1] = +1
        self.wl[3][0] = self.wo_d4; self.sl[3][0] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        # dt e
        self.wl[4][1] = self.wo_d2; self.sl[4][1] = +0
        self.wl[4][0] = self.wc_ml; self.sl[4][0] = +1
        self.wl[4][2] = self.wo_ml; self.sl[4][2] = +1

        self.wl[5][1] = self.wo_d2; self.sl[5][1] = +0
        self.wl[5][0] = self.wc_ml; self.sl[5][0] = -1
        self.wl[5][2] = self.wo_ml; self.sl[5][2] = -1

        self.npad_conv = 2

class qest_plm_TB(qest):
    def __init__(self, clte):
        self.clte = clte
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_di; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_mi; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_mi; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_di; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2

    def wo_di(self, l, lx, ly):
        return +0.25j
    def wo_mi(self, l, lx, ly):
        return -0.25j
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * l

class qest_plm_BT(qest_plm_TB):
    def __init__(self, clte):
        self.clte = clte
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][0] = self.wo_di; self.sl[0][0] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][0] = self.wo_mi; self.sl[1][0] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][1] = self.wc_ml; self.sl[2][1] = -1
        self.wl[2][0] = self.wo_mi; self.sl[2][0] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][1] = self.wc_ml; self.sl[3][1] = +1
        self.wl[3][0] = self.wo_di; self.sl[3][0] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2

class qest_plm_EE(qest):
    def __init__(self, clee):
        self.clee = clee
        self.ntrm = 8

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wo_d4; self.sl[0][0] = -2
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wo_d4; self.sl[1][0] = +2
        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = +3
        self.wl[2][1] = self.wo_d4; self.sl[2][1] = -2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = -3
        self.wl[3][1] = self.wo_d4; self.sl[3][1] = +2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.wl[4][0] = self.wo_d4; self.sl[4][0] = +2
        self.wl[4][1] = self.wc_ml; self.sl[4][1] = -1
        self.wl[4][2] = self.wo_ml; self.sl[4][2] = +1

        self.wl[5][0] = self.wo_d4; self.sl[5][0] = -2
        self.wl[5][1] = self.wc_ml; self.sl[5][1] = +1
        self.wl[5][2] = self.wo_ml; self.sl[5][2] = -1

        self.wl[6][0] = self.wc_ml; self.sl[6][0] = -1
        self.wl[6][1] = self.wo_d4; self.sl[6][1] = +2
        self.wl[6][2] = self.wo_ml; self.sl[6][2] = +1

        self.wl[7][0] = self.wc_ml; self.sl[7][0] = +1
        self.wl[7][1] = self.wo_d4; self.sl[7][1] = -2
        self.wl[7][2] = self.wo_ml; self.sl[7][2] = -1

        self.npad_conv = 2

    def wo_d4(self, l, lx, ly):
        return -0.25
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * l

class qest_plm_EB(qest):
    def __init__(self, clee):
        self.clee = clee
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_di; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_mi; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_mi; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_di; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2

    def wo_di(self, l, lx, ly):
        return +0.25j
    def wo_mi(self, l, lx, ly):
        return -0.25j
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * l

class qest_plm_BE(qest_plm_EB):
    def __init__(self, clee):
        self.clee = clee
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][0] = self.wo_di; self.sl[0][0] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][0] = self.wo_mi; self.sl[1][0] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][1] = self.wc_ml; self.sl[2][1] = -1
        self.wl[2][0] = self.wo_mi; self.sl[2][0] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][1] = self.wc_ml; self.sl[3][1] = +1
        self.wl[3][0] = self.wo_di; self.sl[3][0] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2

class qest_xlm_EB(qest):
    def __init__(self, clee):
        self.clee = clee
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_di; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_di; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_mi; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_mi; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2

    def wo_di(self, l, lx, ly):
        return -0.25
    def wo_mi(self, l, lx, ly):
        return +0.25
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * l

class qest_blm_EP(qest):
    def __init__(self, clee, clpp):
        self.clee = clee
        self.clpp = clpp
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wp_ml; self.sl[0][1] = -1
        self.wl[0][2] = self.wo_di; self.sl[0][2] = +2

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wp_ml; self.sl[1][1] = +1
        self.wl[1][2] = self.wo_mi; self.sl[1][2] = -2

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wp_ml; self.sl[2][1] = -1
        self.wl[2][2] = self.wo_mi; self.sl[2][2] = -2

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wp_ml; self.sl[3][1] = +1
        self.wl[3][2] = self.wo_di; self.sl[3][2] = +2

        self.npad_conv = 2

    def wo_di(self, l, lx, ly):
        return -0.25j
    def wo_mi(self, l, lx, ly):
        return +0.25j
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * l
    def wp_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clpp)), self.clpp, right=0 ) * l

class qest_blm_EX(qest):
    def __init__(self, clee, clpp):
        self.clee = clee
        self.clpp = clpp
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wp_ml; self.sl[0][1] = -1
        self.wl[0][2] = self.wo_di; self.sl[0][2] = +2

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wp_ml; self.sl[1][1] = +1
        self.wl[1][2] = self.wo_di; self.sl[1][2] = -2

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wp_ml; self.sl[2][1] = -1
        self.wl[2][2] = self.wo_mi; self.sl[2][2] = -2

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wp_ml; self.sl[3][1] = +1
        self.wl[3][2] = self.wo_mi; self.sl[3][2] = +2

        self.npad_conv = 2

    def wo_di(self, l, lx, ly):
        return -0.25
    def wo_mi(self, l, lx, ly):
        return +0.25
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * l
    def wp_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clpp)), self.clpp, right=0 ) * l

class qest_tlm_TP(qest):
    def __init__(self, cltt, clpp):
        self.cltt = cltt
        self.clpp = clpp
        self.ntrm = 2

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +1
        self.wl[0][1] = self.wp_ml; self.sl[0][1] = -1
        self.wl[0][2] = self.wo_d2; self.sl[0][2] = +0

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -1
        self.wl[1][1] = self.wp_ml; self.sl[1][1] = +1
        self.wl[1][2] = self.wo_d2; self.sl[1][2] = +0

        self.npad_conv = 2

    def wo_d2(self, l, lx, ly):
        return 1.
    def wo_ml(self, l, lx, ly):
        return l
    def wp_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clpp)), self.clpp, right=0 ) * l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * l


class qest_tau_TT(qest):
    def __init__(self, cltt):
        self.cltt = cltt
        self.ntrm = 2

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc     ; self.sl[0][0] = +0
        self.wl[0][1] = self.wo_p1  ; self.sl[0][1] = +0
        self.wl[0][2] = self.wo_m1  ; self.sl[0][2] = +0

        self.wl[1][0] = self.wo_p1  ; self.sl[1][0] = +0
        self.wl[1][1] = self.wc     ; self.sl[1][1] = +0
        self.wl[1][2] = self.wo_m1  ; self.sl[1][2] = +0

    def wo_p1(self, l, lx, ly):
        return +1.0
    def wo_m1(self, l, lx, ly):
        return -1.0
    def wc(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 )



class qest_tau_EB(qest):
    def __init__(self, clee):
        self.clee = clee
        self.ntrm = 2

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc     ; self.sl[0][0] = -2
        self.wl[0][1] = self.wo_p1  ; self.sl[0][1] = +2
        self.wl[0][2] = self.wo_aa  ; self.sl[0][2] = +0

        self.wl[1][0] = self.wc     ; self.sl[1][0] = +2
        self.wl[1][1] = self.wo_m1  ; self.sl[1][1] = -2
        self.wl[1][2] = self.wo_aa  ; self.sl[1][2] = +0

    def wo_p1(self, l, lx, ly):
        return +1.0
    def wo_m1(self, l, lx, ly):
        return -1.0
    def wo_aa(self, l, lx, ly):
        return 1.0/(2j)
        #return -1.0/(2j)
    def wc(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 )



class qest_tau_TB(qest):
    def __init__(self, clte):
        self.clte = clte
        self.ntrm = 2

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc     ; self.sl[0][0] = -2
        self.wl[0][1] = self.wo_p1  ; self.sl[0][1] = +2
        self.wl[0][2] = self.wo_aa  ; self.sl[0][2] = +0

        self.wl[1][0] = self.wc     ; self.sl[1][0] = +2
        self.wl[1][1] = self.wo_m1  ; self.sl[1][1] = -2
        self.wl[1][2] = self.wo_aa  ; self.sl[1][2] = +0

    def wo_p1(self, l, lx, ly):
        return +1.0
    def wo_m1(self, l, lx, ly):
        return -1.0
    def wo_aa(self, l, lx, ly):
        return 1.0/(2j)
        #return -1.0/(2j)
    def wc(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 )


class qest_tau_EE(qest):
    def __init__(self, clee):
        self.clee = clee
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc     ; self.sl[0][0] = -2
        self.wl[0][1] = self.wo_p1  ; self.sl[0][1] = +2
        self.wl[0][2] = self.wo_aa  ; self.sl[0][2] = +0

        self.wl[1][0] = self.wc     ; self.sl[1][0] = +2
        self.wl[1][1] = self.wo_p1  ; self.sl[1][1] = -2
        self.wl[1][2] = self.wo_aa  ; self.sl[1][2] = +0

        self.wl[2][0] = self.wo_p1  ; self.sl[2][0] = -2
        self.wl[2][1] = self.wc     ; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_aa  ; self.sl[2][2] = +0

        self.wl[3][0] = self.wo_p1  ; self.sl[3][0] = +2
        self.wl[3][1] = self.wc     ; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_aa  ; self.sl[3][2] = +0

    def wo_p1(self, l, lx, ly):
        return +1.0
    def wo_aa(self, l, lx, ly):
        return -0.5
    def wc(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 )


class qest_tau_TE(qest):
    def __init__(self, clte):
        self.clte = clte
        self.ntrm = 3

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc     ; self.sl[0][0] = -2
        self.wl[0][1] = self.wo_p1  ; self.sl[0][1] = +2
        self.wl[0][2] = self.wo_aa  ; self.sl[0][2] = +0

        self.wl[1][0] = self.wc     ; self.sl[1][0] = +2
        self.wl[1][1] = self.wo_p1  ; self.sl[1][1] = -2
        self.wl[1][2] = self.wo_aa  ; self.sl[1][2] = +0

        self.wl[2][0] = self.wo_p1  ; self.sl[2][0] = +0
        self.wl[2][1] = self.wc     ; self.sl[2][1] = +0
        self.wl[2][2] = self.wo_m1  ; self.sl[2][2] = +0

    def wo_p1(self, l, lx, ly):
        return +1.0
    def wo_m1(self, l, lx, ly):
        return -1.0
    def wo_aa(self, l, lx, ly):
        return -0.5
    def wc(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 )
