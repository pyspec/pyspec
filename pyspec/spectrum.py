import numpy as np
try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

class Spectrum(object):
    """ A class that represents the a single realization of 
            the one-dimensional spectrum  of a given field phi """
    
    def __init__(self,phi,dt):

        self.phi = phi      # field to be analyzed
        self.dt = dt        # sampling interval
        self.n = phi.size 

        # test if n is even
        if (self.n%2):
            self.neven = False
        else:
            self.neven = True
    
        # calculate frequencies
        self.calc_freq()

        # calculate spectrum
        self.calc_spectrum()

        # calculate total var
        self.calc_var()

    def calc_freq(self):
        """ calculate array of spectral variable (frequency or 
                wavenumber) in cycles per unit of L """

        self.df = 1./((self.n-1)*self.dt)

        if self.neven:
            self.f = self.df*np.arange(self.n/2+1)
        else:
            self.f = self.df*np.arange( (self.n-1)/2.  + 1 )
            
    def calc_spectrum(self):
        """ compute the 1d spectrum of a field phi """

        self.phih = np.fft.rfft(self.phi)

        # the factor of 2 comes from the symmetry of the Fourier coeffs
        self.spec = 2.*(self.phih*self.phih.conj()).real / self.df /\
                self.n**2

        # the zeroth frequency should be counted only once
        self.spec[0] = self.spec[0]/2.
        if self.neven:
            self.spec[-1] = self.spec[-1]/2.

    def calc_var(self):
        """ Compute total variance from spectrum """
        self.var = self.df*self.spec[1:].sum()  # do not consider zeroth frequency





