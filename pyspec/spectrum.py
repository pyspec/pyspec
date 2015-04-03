import numpy as np
try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

pi = np.pi

class Spectrum(object):
    """ A class that represents the one-dimensional spectrum """
    
    def __init__(self,phi,L):

        self.phi = phi
        self.L = L
        self.n = phi.size
    
        # calculate frequencies
        self.calc_freq()

        # calculate spectrum
        self.calc_spectrum()

    def calc_freq(self):
        """ calculate array of spectral variable (frequency or 
                wavenumber) in cycles per unit of L """

        self.df = 1./self.L

        if (self.n%2):
            self.f = self.df*np.arange(self.n/2+1)
        else:
            self.f = self.df*np.arange( (self.n-1) / 2*self.n + 1)
            
    def calc_spectrum(self):
        """ compute the 1d spectrum of a field phi """

        self.phih = np.fft.rfft(self.phi)
        self.spec = 2.*(self.phih*self.phih.conj()).real / self.df /\
                self.n**2

        # consider only 1 zeroth and nyquist frequency
        self.spec[-1] = self.spec[-1]/2.
        self.spec[0] = self.spec[0]/2.

