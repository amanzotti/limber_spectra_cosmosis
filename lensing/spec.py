import os, copy, glob
import numpy as np
import util
import maps

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

def get_camb_scalcl(prefix=None, lmax=None, fname=None):
    """ loads and returns a "scalar Cls" file produced by CAMB (camb.info).
    can either use a prefix indicating one of the Cl files included in lensing/data, or a full filename.
    lmax sets maximum multipole to load (all multipoles will be loaded by default). """
    if fname == None:
        basedir = os.path.dirname(__file__)

        if prefix == None:
            prefix = "ebp_lcdm"
        fname = basedir + "/data/cl/" + prefix + "/*_scalCls.dat"

    tf = glob.glob( fname )
    assert(len(tf) == 1)

    return camb_clfile( tf[0], lmax=lmax )

def get_camb_lensedcl(prefix=None, lmax=None, fname=None):
    """ loads and returns a "lensed Cls" file produced by CAMB (camb.info).
    can either use a prefix indicating one of the Cl files included in lensing/data, or a full filename.
    lmax sets maximum multipole to load (all multipoles will be loaded by default). """
    if fname ==None:
        basedir = os.path.dirname(__file__)

        if prefix == None:
            prefix = "ebp_lcdm"
        fname = basedir + "/data/cl/" + prefix + "/*_lensedCls.dat"

    tf = glob.glob( fname )
    assert(len(tf) == 1)
    return camb_clfile( tf[0], lmax=lmax )

def is_camb_clfile(object):
    """ ducktyping check of whether the given object is a set of Cls produced by CAMB """
    if not hasattr(object, 'lmax'):
        return False
    if not hasattr(object, 'ls'):
        return False
    return set(object.__dict__.keys()).issubset( set( ['lmax', 'ls', 'cltt', 'clee', 'clte', 'clpp', 'cltp', 'clbb', 'clep', 'cleb' ] ) )

class camb_clfile(object):
    """ class to hold Cls loaded from a the output files produced by CAMB. """
    def __init__(self, tfname, lmax=None):
        """ load Cls from an _scalarCls or _lensedCls file given by tfname.
        lmax = maximum multipole to load (if None then all multipoles will be loaded. """
        tarray = np.loadtxt(tfname)
        lmin   = tarray[0, 0]
        assert(lmin in [1,2])

        if lmax == None:
            lmax = np.shape(tarray)[0]-lmin+1
            assert(tarray[-1, 0] == lmax)
        assert( (np.shape(tarray)[0]+1) >= lmax )

        ncol = np.shape(tarray)[1]
        ell  = np.arange(lmin, lmax+1, dtype=np.float)

        self.lmax = lmax
        self.ls   = np.concatenate( [ np.arange(0, lmin), ell ] )
        if ncol == 5:
            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.)       ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.)        ] )
            self.clbb = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.)        ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]*2.*np.pi/ell/(ell+1.)        ] )

        elif ncol == 6:
            tcmb  = 2.726*1e6 #uK

            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.)       ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.)       ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.)       ] )
            self.clpp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]/ell**4/tcmb**2     ] )
            self.cltp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),5]/ell**3/tcmb ] )

    def copy(self, lmax=None, lmin=None):
        """ clone this object, optionally restricting to the multipole range given by [lmin, lmax] """
        if (lmax == None):
            return copy.deepcopy(self)
        else:
            assert( lmax <= self.lmax )
            ret      = copy.deepcopy(self)
            ret.lmax = lmax
            ret.ls   = np.arange(0, lmax+1)
            for k, v in self.__dict__.items():
                if k[0:2] == 'cl':
                    setattr( ret, k, copy.deepcopy(v[0:lmax+1]) )

            if lmin != None:
                assert( lmin <= lmax )
                for k in self.__dict__.keys():
                    if k[0:2] == 'cl':
                        getattr( ret, k )[0:lmin] = 0.0
            return ret

    def hashdict(self):
        """ return a dictionary uniquely associated with the contents of this clfile. """
        ret = {}
        for attr in ['lmax', 'cltt', 'clee', 'clte', 'clpp', 'cltp', 'clbb', 'clep', 'cleb' ]:
            if hasattr(self, attr):
                ret[attr] = getattr(self, attr)
        return ret

    def __add__(self, other):
        if is_camb_clfile(other):
            assert( self.lmax == other.lmax )
            ret = self.copy()
            zs  = np.zeros(self.lmax+1)
            for attr in ['cltt', 'clee', 'clte', 'clpp', 'cltp', 'clbb', 'clep', 'cleb' ]:
                if (hasattr(self, attr) or hasattr(other, attr)):
                    setattr(ret, attr, getattr(self, attr, zs) + getattr(other, attr, zs) )
            return ret
        else:
            assert(0)

    def __eq__(self, other):
        try:
            for key in self.__dict__.keys()+other.__dict__.keys():
                if type(self.__dict__[key]) == np.ndarray:
                    assert( np.all( self.__dict__[key] == other.__dict__[key] ) )
                else:
                    assert( self.__dict__[key] == other.__dict__[key] )
        except:
            return False

        return True

    # def plot(self, spec='cltt', p=pl.plot, t=lambda l:1., **kwargs):
    #     """ plot the spectrum
    #          * spec - spectrum to display (e.g. cltt, clee, clte, etc.)
    #          * p    - plotting function to use p(x,y,**kwargs)
    #          * t    - scaling to apply to the plotted Cl -> t(l)*Cl
    #     """
    #     p( self.ls, t(self.ls) * getattr(self, spec), **kwargs )

def is_lcl(obj):
    """ ducktyping check of whether this is an lcl object. """
    return ( hasattr(obj, 'lmax') and hasattr(obj, 'dl') and
             hasattr(obj, 'nm') and hasattr(obj, 'cl') )

class lcl(object):
    """
    Class to hold 1-dimensional power spectra with uniform binning. Contains attributes:
       dl,          "delta ell," the size of binning
       lmax,        the maximum ell to keep track of
       cl,          the power spectrum value Cl
       nm,          number of modes (number or pixels in each annulus)
    """
    def __init__(self, lmax, fc, dl=1):
        """
        Constructor.
          fc,         complex FFT to that we want to calculate the power spectrum from
        """
        self.lmax = lmax
        self.dl   = dl

        ell = fc.get_ell().flatten()
        self.nm, bins = np.histogram(ell, bins=np.arange(0, lmax+1, dl))
        self.cl, bins = np.histogram(ell, bins=np.arange(0, lmax+1, dl), weights=fc.fft.flatten())
        self.cl[np.nonzero(self.nm)] /= self.nm[np.nonzero(self.nm)]

    def is_compatible(self, other):
        """ Check if this object can be added, subtracted, etc. with other. """
        return ( is_lcl(other) and (self.lmax == other.lmax) and (self.dl == other.dl) and np.all(self.nm == other.nm) )

    def __add__(self, other):
        assert( self.is_compatible(other) )
        ret = copy.deepcopy(self)
        ret.cl += other.cl
        return ret
    def __sub__(self, other):
        assert( self.is_compatible(other) )
        ret = copy.deepcopy(self)
        ret.cl -= other.cl
        return ret
    def __mul__(self, other):
        if np.isscalar(other):
            ret = copy.deepcopy(self)
            ret.cl *= other
            return ret
        elif is_lcl(other):
            assert( self.is_compatible(other) )
            ret = copy.deepcopy(self)
            ret.cl[np.nonzero(self.nm)] *= other.cl[np.nonzero(self.nm)]
            return ret
        else:
            assert(0)
    def __div__(self, other):
        if np.isscalar(other):
            ret = copy.deepcopy(self)
            ret.cl /= other
            return ret
        elif is_lcl(other):
            assert( self.is_compatible(other) )
            ret = copy.deepcopy(self)
            ret.cl[np.nonzero(self.nm)] /= other.cl[np.nonzero(self.nm)]
            return ret
        else:
            assert(0)
    def get_ml(self, lbins, w=lambda l : 1.):
        """
        Rebins this spectrum with non-uniform binning as a bcl object.
          * lbins - numpy-array definining the bin edges [lbins[0], lbins[1]], [lbins[1], lbins[2]], ...
          * w     - weight function to apply when accumulating into bins.
        """
        lb = 0.5*(lbins[:-1] + lbins[1:])
        wb = w(lb)

        l = np.arange(0, self.lmax+1, self.dl)
        l = 0.5*(l[:-1] + l[1:]) # get bin centers
        w = w(l)

        norm, bins = np.histogram(l, bins=lbins, weights=np.nan_to_num(self.nm)) # Get weights in each l-bin
        spec, bins = np.histogram(l, bins=lbins, weights=w*np.nan_to_num(self.nm)*np.nan_to_num(self.cl)) # Bin the spectrum

        spec[np.nonzero(norm)] /= norm[np.nonzero(norm)]*wb # normalize the spectrum

        return bcl(lbins, {'cl' : spec})

class bcl(object):
    """
    Binned power spectrum. Contains attributes:
      specs,           dictionary, contaning binned spectra
      lbins,           list defining the bin edges.
      ls,              bin centers, given by average of left and right edges.
    """
    def __init__(self, lbins, specs):
        self.lbins = lbins
        self.specs = specs

        self.ls    = 0.5*(lbins[0:-1] + lbins[1:]) # get bin centers

    def __getattr__(self, spec):
        try:
            return self.specs[spec]
        except KeyError:
            raise AttributeError(spec)

    def __mul__(self, other):
        ret = copy.deepcopy(self)

        if np.isscalar(other):
            for spec in ret.specs.keys():
                ret.specs[spec][:] *= other
        elif (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )
            for spec in ret.specs.keys():
                ret.specs[spec][:] *= other.specs[spec][:]
        else:
            assert(0)

        return ret

    def __div__(self, other):
        ret = copy.deepcopy(self)

        if np.isscalar(other):
            for spec in ret.specs.keys():
                ret.specs[spec][:] /= other
        elif (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )
            for spec in ret.specs.keys():
                ret.specs[spec][:] /= other.specs[spec][:]
        else:
            assert(0)

        return ret

    def __add__(self, other):
        if (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )

            ret = copy.deepcopy(self)
            for spec in ret.specs.keys():
                ret.specs[spec][:] += other.specs[spec][:]
            return ret
        else:
            assert(0)

    def __sub__(self, other):
        if (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )

            ret = copy.deepcopy(self)
            for spec in ret.specs.keys():
                ret.specs[spec][:] -= other.specs[spec][:]
            return ret
        else:
            assert(0)

    def __iadd__(self, other):
        if (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )
            assert( self.specs.keys() == other.specs.keys() )

            for spec in self.specs.keys():
                self.specs[spec][:] += other.specs[spec]
        else:
            assert(0)

        return self

    def get_ml(self, lbins, w=lambda l : 1.):
        """ Return the average cl in annuli defined by the list of bin edges lbins.
        Currently only implemented for trivial case where lbins are the same as those used by this object. """
        if np.all(self.lbins == lbins):
            return self
        else:
            assert(0)

    # def plot(self, spec='cl', p=pl.plot, t=lambda l:1., **kwargs):
    #     """ plot the binned spectrum
    #          * spec - spectrum to display (e.g. cltt, clee, clte, etc.)
    #          * p    - plotting function to use p(x,y,**kwargs)
    #          * t    - scaling to apply to the plotted Cl -> t(l)*Cl
    #     """
    #     p( self.ls, t(self.ls) * self.specs[spec], **kwargs )

    def inverse(self):
        ret = copy.deepcopy(self)
        for spec in ret.specs.keys():
            ret.specs[spec][:] = 1./ret.specs[spec][:]
        return ret

def is_clmat_t(obj):
    if not ( hasattr(obj, 'lmax') and hasattr(obj, 'clmat') ):
        return False
    return obj.clmat.shape == (obj.lmax+1)

class clmat_t(object):
    def __init__(self, clmat):
        self.lmax  = len(clmat)-1
        self.clmat = clmat.copy()

    def clone(self, lmax=None):
        if lmax == None:
            lmax = self.lmax
        assert(lmax <= self.lmax)
        ret = sinv_filt( np.zeros(lmax+1) )
        ret.clmat[:] = self.clmat[0:lmax+1]

        return ret

    def __add__(self, other):
        if ( hasattr(other, 'fft') and hasattr(other, 'get_ell') ):
            ret = other.copy()
            ell = other.get_ell()

            ret.fft[:,:]  += np.interp( ell.flatten(), np.arange(0, self.lmax+1), self.clmat[:], right=0 ).reshape(ell.shape)
            return ret
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            assert( (self.lmax+1) <= len(other) )
            return clmat_t( self.clmat + other[0:self.lmax+1] )
        else:
            assert(0)

    def __mul__(self, other):
        if False:
            pass
        elif np.isscalar(other):
            return clmat_t( self.clmat * other )
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            assert( (self.lmax+1) <= len(other) )
            return clmat_t( self.clmat * other[0:lmax+1] )

        elif ( hasattr(other, 'fft') and hasattr(other, 'get_ell') ):
            ret = other.copy()
            ell = other.get_ell()

            def fftxcl(fft, cl):
                return fft * np.interp( ell.flatten(), np.arange(0, len(cl)), cl, right=0 ).reshape(fft.shape)

            ret.fft[:,:]  = fftxcl( other.fft, self.clmat[:] )
            return ret
        else:
            assert(0)

    def inverse(self):
        ret = clmat_t( np.zeros(self.lmax+1) )
        ret.clmat[ np.nonzero(self.clmat) ] = 1./self.clmat[ np.nonzero(self.clmat) ]
        return ret

    def cholesky(self):
        return clmat_t( np.sqrt(self.clmat) )

def is_clmat_teb(obj):
    if not ( hasattr(obj, 'lmax') and hasattr(obj, 'clmat') ):
        return False
    return obj.clmat.shape == (obj.lmax+1, 3, 3)

class clmat_teb(object):
    """ Class to hold the 3x3 covariance matrix at each multipole for a set of T, E and B auto- and cross-spectra. """
    def __init__(self, cl):
        """ Initializes this clmat_teb object using the power spectra cl.cltt, cl.clte, clee, etc.
        Spectra which are not present in cl are assumed to be zero.
        """
        lmax = cl.lmax
        zs   = np.zeros(lmax+1)

        clmat = np.zeros( (lmax+1, 3, 3) ) # matrix of TEB correlations at each l.
        clmat[:,0,0] = getattr(cl, 'cltt', zs.copy())
        clmat[:,0,1] = getattr(cl, 'clte', zs.copy()); clmat[:,1,0] = clmat[:,0,1]
        clmat[:,0,2] = getattr(cl, 'cltb', zs.copy()); clmat[:,2,0] = clmat[:,0,2]
        clmat[:,1,1] = getattr(cl, 'clee', zs.copy())
        clmat[:,1,2] = getattr(cl, 'cleb', zs.copy()); clmat[:,2,1] = clmat[:,1,2]
        clmat[:,2,2] = getattr(cl, 'clbb', zs.copy())

        self.lmax  = lmax
        self.clmat = clmat

    def hashdict(self):
        return { 'lmax' : self.lmax,
                 'clmat': hashlib.md5( self.clmat.view(np.uint8) ).hexdigest() }

    def compatible(self, other):
        """ Test whether this object and the clmat_teb object other can be added, subtracted, or multiplied. """
        return ( ( self.lmax == other.lmax ) and
                 ( self.clmat.shape == other.clmat.shape ) )

    # def clone(self, lmax=None):
    #     if lmax == None:
    #         lmax = self.lmax
    #     ret = clmat_teb( util.dictobj( { 'lmax' : lmax } ) )
    #     ret.clmat[:,:,:] = self.clmat[0:lmax+1,:,:]

        return ret

    def __add__(self, other):
        if is_clmat_teb(other):
            assert( self.compatible(other) )
            ret = copy.deepcopy(self)
            ret.clmat += other.clmat
            return ret
        elif maps.is_tebfft(other):
            teb = other
            ret = teb.copy()
            ell = teb.get_ell()

            ret.tfft[:,:]  += np.interp( ell.flatten(), np.arange(0, len(self.clmat[:,0,0])), self.clmat[:,0,0], right=0 ).reshape(ell.shape)
            ret.efft[:,:]  += np.interp( ell.flatten(), np.arange(0, len(self.clmat[:,1,1])), self.clmat[:,1,1], right=0 ).reshape(ell.shape)
            ret.bfft[:,:]  += np.interp( ell.flatten(), np.arange(0, len(self.clmat[:,2,2])), self.clmat[:,2,2], right=0 ).reshape(ell.shape)

            return ret
        else:
            assert(0)

    def __mul__(self, other):
        if False:
            pass
        elif np.isscalar(other):
            ret = self.clone()
            ret.clmat *= other
            return ret
        elif is_clmat_teb(other):
            assert( self.compatible(other) )
            ret = self.clone()
            ret.clmat *= other.clmat
            return ret
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            lmax = self.lmax
            assert( (lmax+1) >= len(other) )

            ret = self.clone()
            for i in xrange(0,3):
                for j in xrange(0,3):
                    ret.clmat[:,i,j] *= other[0:lmax+1]

            return ret

        elif ( hasattr(other, 'tfft') and hasattr(other, 'efft') and hasattr(other, 'bfft') and hasattr(other, 'get_ell') ):
            teb = other
            ret = teb.copy()
            ell = teb.get_ell()

            def fftxcl(fft, cl):
                return fft * np.interp( ell.flatten(), np.arange(0, len(cl)), cl, right=0 ).reshape(fft.shape)

            ret.tfft[:,:]  = fftxcl( teb.tfft, self.clmat[:,0,0] ) + fftxcl( teb.efft, self.clmat[:,0,1] ) + fftxcl( teb.bfft, self.clmat[:,0,2] )
            ret.efft[:,:]  = fftxcl( teb.tfft, self.clmat[:,1,0] ) + fftxcl( teb.efft, self.clmat[:,1,1] ) + fftxcl( teb.bfft, self.clmat[:,1,2] )
            ret.bfft[:,:]  = fftxcl( teb.tfft, self.clmat[:,2,0] ) + fftxcl( teb.efft, self.clmat[:,2,1] ) + fftxcl( teb.bfft, self.clmat[:,2,2] )

            return ret
        else:
            assert(0)

    def inverse(self):
        """ Return a new clmat_teb object, which contains the matrix inverse of this one, multipole-by-multipole. """
        ret = copy.deepcopy(self)
        for l in xrange(0, self.lmax+1):
            ret.clmat[l,:,:] = np.linalg.pinv( self.clmat[l] )
        return ret

    def cholesky(self):
        """ Return a new clmat_teb object, which contains the cholesky decomposition (or matrix square root) of this one, multipole-by-multipole. """
        ret = copy.deepcopy(self)
        for l in xrange(0, self.lmax+1):
            u, t, v = np.linalg.svd(self.clmat[l])
            ret.clmat[l,:,:] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
        return ret

# class blmat_teb(clmat_teb):
#     """ Special case of clmat_teb which is diagonal, with the TT, EE, and BB covariances all equal to B(l). """
#     def __init__(self, bl):
#         super(blmat_teb, self).__init__( util.dictobj( { 'lmax' : len(bl)-1,
#                                                          'cltt' : bl,
#                                                          'clee' : bl,
#                                                          'clbb' : bl } ) )

# def tebfft2cl( lbins, teb1, teb2=None, w=None,  psimin=0., psimax=np.inf, psispin=1  ):
#     """ Calculate the annulus-averaged auto- or cross-spectrum of teb object(s),
#     for bins described by lbins and a weight function w=w(l).
#             * lbins   = list of bin edges.
#             * teb1    = tebfft object.
#             * teb2    = optional second tebfft object to cross-correlate with teb1 (otherwise will return an auto-spectrum).
#             * w       = function w(l) which weights the FFT when averaging.
#             * psimin, psimax, psispin = parameters used to set wedges for the annular average.
#                    psi = mod(psispin * arctan2(lx, -ly), 2pi) in the range [psimin, psimax].
#     """
#     dopsi = ( (psimin, psimax, psispin) != (0., np.inf, 1) )

#     if teb2 == None:
#         teb2 = teb1

#     assert( teb1.compatible( teb2 ) )

#     ell = teb1.get_ell().flatten()
#     if dopsi:
#         lx, ly = teb1.get_lxly()
#         psi = np.mod( psispin*np.arctan2(lx, -ly), 2.*np.pi ).flatten()

#     if w == None:
#         w = np.ones(ell.shape)
#     else:
#         w = w(ell)

#     if dopsi:
#         w[ np.where( psi < psimin ) ] = 0.0
#         w[ np.where( psi >= psimax ) ] = 0.0

#     norm, bins = np.histogram(ell, bins=lbins, weights=w); norm[ np.where(norm != 0.0) ] = 1./norm[ np.where(norm != 0.0) ]
#     cltt, bins = np.histogram(ell, bins=lbins, weights=w*(teb1.tfft * np.conj(teb2.tfft)).flatten().real); cltt *= norm
#     clte, bins = np.histogram(ell, bins=lbins, weights=w*(teb1.tfft * np.conj(teb2.efft)).flatten().real); clte *= norm
#     cltb, bins = np.histogram(ell, bins=lbins, weights=w*(teb1.tfft * np.conj(teb2.bfft)).flatten().real); cltb *= norm
#     clee, bins = np.histogram(ell, bins=lbins, weights=w*(teb1.efft * np.conj(teb2.efft)).flatten().real); clee *= norm
#     cleb, bins = np.histogram(ell, bins=lbins, weights=w*(teb1.efft * np.conj(teb2.bfft)).flatten().real); cleb *= norm
#     clbb, bins = np.histogram(ell, bins=lbins, weights=w*(teb1.bfft * np.conj(teb2.bfft)).flatten().real); clbb *= norm

#     return bcl(lbins, { 'cltt' : cltt,
#                         'clte' : clte,
#                         'cltb' : cltb,
#                         'clee' : clee,
#                         'cleb' : cleb,
#                         'clbb' : clbb } )

# def cross_cl( lbins, r1, r2=None, w=None ):
#     """ Returns the auto- or cross-spectra of either rfft or tebfft objects. """
#     if r2 is None:
#         r2 = r1

#     assert( r1.compatible( r2 ) )

#     if maps.is_tebfft(r1):
#         return tebfft2cl(lbins, r1, r2, w=w)
#     elif maps.is_rfft(r1):
#         return rcfft2cl(lbins, r1, r2, w=w)
#     elif maps.is_cfft(r1):
#         return rcfft2cl(lbins, r1, r2, w=w)
#     else:
#         assert(0)

# def rcfft2cl( lbins, r1, r2=None, w=None, psimin=0., psimax=np.inf, psispin=1 ):
#     """ Calculate the annulus-averaged auto- or cross-spectrum of rfft object(s),
#     for bins described by lbins and a weight function w=w(l).
#             * lbins   = list of bin edges.
#             * r1      = tebfft object.
#             * r2      = optional second rfft object to cross-correlate with teb1 (otherwise will return an auto-spectrum).
#             * w       = function w(l) which weights the FFT when averaging.
#             * psimin, psimax, psispin = parameters used to set wedges for the annular average.
#                    psi = mod(psispin * arctan2(lx, -ly), 2pi) in the range [psimin, psimax].
#     """
#     dopsi = ( (psimin, psimax, psispin) != (0., np.inf, 1) )
#     if r2 is None:
#         r2 = r1

#     assert( r1.compatible( r2 ) )

#     ell = r1.get_ell().flatten()

#     if dopsi:
#         lx, ly = r1.get_lxly()
#         psi = np.mod( psispin*np.arctan2(lx, -ly), 2.*np.pi ).flatten()

#     if w == None:
#         w = np.ones( ell.shape )
#     else:
#         w = w(ell)

#     c = (r1.fft * np.conj(r2.fft)).flatten()
#     w[ np.isnan(c) ] = 0.0
#     c[ np.isnan(c) ] = 0.0

#     if dopsi:
#         w[ np.where( psi < psimin ) ] = 0.0
#         w[ np.where( psi >= psimax ) ] = 0.0

#     norm, bins = np.histogram(ell, bins=lbins, weights=w); norm[ np.where(norm != 0.0) ] = 1./norm[ np.where(norm != 0.0) ]
#     clrr, bins = np.histogram(ell, bins=lbins, weights=w*c); clrr *= norm

#     return bcl(lbins, { 'cl' : clrr } )

# def cl2cfft(cl, pix):
#     """ Returns a cfft object, with FFT(lx,ly) = cl[ sqrt(lx**2 + ly**2]) ] """
#     ell = pix.get_ell().flatten()

#     ret = maps.cfft( nx=pix.nx, dx=pix.dx,
#                      fft=np.array( np.interp( ell, np.arange(0, len(cl)), cl, right=0 ).reshape(pix.nx, pix.ny), dtype=np.complex ),
#                      ny=pix.ny, dy=pix.dy )

#     return ret

# def cl2tebfft(cl, pix):
#     ell = pix.get_ell().flatten()

#     return maps.tebfft( pix.nx, pix.dx,
#                         ffts=[ np.array( np.interp( ell, np.arange(0, cl.lmax+1), cl.cltt, right=0 ).reshape(pix.ny, pix.nx/2+1), dtype=np.complex ),
#                                np.array( np.interp( ell, np.arange(0, cl.lmax+1), cl.clee, right=0 ).reshape(pix.ny, pix.nx/2+1), dtype=np.complex ),
#                                np.array( np.interp( ell, np.arange(0, cl.lmax+1), cl.clbb, right=0 ).reshape(pix.ny, pix.nx/2+1), dtype=np.complex ) ],
#                         ny=pix.ny, dy=pix.dy )

# def plot_cfft_cl2d( cfft, cfft2=None, smth=0, lcnt=None, cm=pl.cm.jet, t = lambda l, v : np.log(np.abs(v)), axlab=True, vmin=None, vmax=None, cbar=False):
#     """ Plot the two-dimensional power spectrum of a tebfft object.
#           * cfft2  = optional second cfft object for cross-correlation.
#           * smth   = gaussian smoothing (in units of pixels) to apply to the 2D spectrum when plotting.
#           * lcnt   = list of L contours to overplot.
#           * cm     = colormap
#           * t      = scaling to apply to each mode as a function of (l)
#           * axlab  = add lx and ly axis labels? (boolean)
#           * vmin   = color scale minimum.
#           * vmax   = color scale maximum.
#           * cbar   = include a colorbar.
#     """
#     if cfft2 is None:
#         cfft2 = cfft

#     assert( cfft.compatible(cfft2) )

#     lx, ly = cfft.get_lxly()

#     lx     = np.fft.fftshift(lx)
#     ly     = np.fft.fftshift(ly)

#     ell    = np.sqrt(lx**2 + ly**2)

#     ext    = [lx[0,0], lx[-1,-1], ly[-1,-1], ly[0,0]]

#     pl.imshow( scipy.ndimage.gaussian_filter( t(ell, np.fft.fftshift( (cfft.fft * np.conj(cfft2.fft)).real ) ), smth),
#                interpolation='nearest', extent=ext, cmap=cm, vmin=vmin, vmax=vmax )
#     if cbar == True:
#         pl.colorbar()
#     pl.contour( lx, ly, ell, levels=lcnt, colors='k', linestyles='--' )

#     if axlab == True:
#         pl.xlabel(r'$\ell_{x}$')
#         pl.ylabel(r'$\ell_{y}$')
