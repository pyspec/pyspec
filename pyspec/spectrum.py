import numpy as np
from numpy import pi
from scipy.special import gammainc
try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

class Spectrum(object):
    """ A class that represents a single realization of 
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

class TWODimensional_spec(object):
    """ A class that represent a two dimensional spectrum 
            for real signals """

    def __init__(self,phi,d1,d2):

        self.phi = phi  # two dimensional real field
        self.d1 = d1
        self.d2 = d2
        self.n1,self.n2 = phi.shape
        self.L1 = d1*self.n1
        self.L2 = d2*self.n2

        # test eveness
        if (self.n1 % 2):
            self.n1even = False
        else: self.n1even = True

        if (self.n2 % 2):
            self.n2even = False
        else: self.n2even = True

        # calculate frequencies
        self.calc_freq()

        # calculate spectrum
        self.calc_spectrum()

        # calculate total var
        self.calc_var()
           
      # calculate isotropic spectrum
        self.calc_ispec()

    def calc_freq(self):
        """ calculate array of spectral variable (frequency or 
                wavenumber) in cycles per unit of L """

        # wavenumber one (equals to dk1 and dk2)
        self.dk1 = 1./self.L1
        self.dk2 = 1./self.L2

        # wavenumber grids
        k2 = self.dk2*np.append( np.arange(0.,self.n2/2), \
            np.arange(-self.n2/2,0.) )
        k1 = self.dk1*np.arange(0.,self.n1/2+1)

        self.kk1,self.kk2 = np.meshgrid(k1,k2)
    
        self.kk1 = np.fft.fftshift(self.kk1,axes=0)
        self.kk2 = np.fft.fftshift(self.kk2,axes=0)
        self.kappa2 = self.kk1**2 + self.kk2**2
        self.kappa = np.sqrt(self.kappa2)

    def calc_spectrum(self):
        """ calculates the spectrum """
        self.phih = np.fft.rfft2(self.phi)
        self.spec = 2.*(self.phih*self.phih.conj()).real/ (self.dk1*self.dk2)\
                / (self.n1*self.n2)**2
        self.spec =  np.fft.fftshift(self.spec,axes=0)

    def calc_var(self):
        """ compute variance of p from Fourier coefficients ph """
        self.var_dens = np.fft.fftshift(self.spec.copy(),axes=0)
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2

        if self.n1even:
            self.var_dens[:,0],self.var_dens[:,-1] = self.var_dens[:,0]/2.,\
                    self.var_dens[:,-1]/2.
            self.var = self.var_dens.sum()*self.dk1*self.dk2
        else:
            self.var_dens[:,0],self.var_dens[:,-1] = self.var_dens[:,0]/2.,\
                    self.var_dens[:,-1]
            self.var = self.var_dens.sum()*self.dk1*self.dk2


    def calc_ispec(self):
        """ calculates isotropic spectrum """
        if self.kk1.max()>self.kk2.max():
            kmax = self.kk2.max()
        else:
            kmax = self.kk1.max()

        # create radial wavenumber
        self.dkr = np.sqrt(self.dk1**2 + self.dk2**2)
        self.kr =  np.arange(self.dkr/2.,kmax+self.dkr,self.dkr)
        self.ispec = np.zeros(self.kr.size)

        for i in range(self.kr.size):
            fkr =  (self.kappa>=self.kr[i]-self.dkr/2) \
                    & (self.kappa<=self.kr[i]+self.dkr/2)
            dth = pi / (fkr.sum()-1)
            self.ispec[i] = self.spec[fkr].sum() * self.kr[i] * dth

def spec_error(E,sn,ci=.95):

    """ Computes confidence interval for one-dimensional spectral 
        estimate E.

        Parameters
        ===========
        - sn is the number of spectral realizations;
                it can be either an scalar or an array of size(E)
        - ci = .95 for 95 % confidence interval

        Output
        ==========
        lower (El) and upper (Eu) bounds on E       """

    dbin = .005
    yN = np.arange(0,2.+dbin,dbin)

    El, Eu = np.empty_like(E), np.empty_like(E)

    try:
        n = sn.size
    except AttributeError:
        n = 0

    if n:
        
        assert n == E.size, " *** sn has different size than E "

        for i in range(n):
            yNl,yNu = yNlu(sn[i],yN=yN,ci=ci)
            El[i] = E[i]/yNl
            Eu[i] = E[i]/yNu
            
    else:
        yNl,yNu = yNlu(sn,yN=yN,ci=ci)
        El = E/yNl
        Eu = E/yNu

    return El, Eu 


def yNlu(sn,yN,ci):
    """ compute yN[l] yN[u], that is, the lower and 
                upper limit of yN """
   
    # cdf of chi^2 dist. with 2*sn DOF
    cdf = gammainc(sn,sn*yN) 

    # indices that delimit the wedge of the conf. interval
    fl = np.abs(cdf - ci).argmin()
    fu = np.abs(cdf - 1. + ci).argmin()

    return yN[fl],yN[fu]


def avg_per_decade(k,E,nbins = 10):
    """ Averages the spectra with nbins per decade

        Parameters
        ===========
        - E is the spectrum
        - k is the original wavenumber array
        - nbins is the number of bins per decade

        Output
        ==========
        - ki: the wavenumber for the averaged spectrum
        - Ei: the averaged spectrum """

    dk = 1./nbins
    logk = np.log10(k)

    logki = np.arange(np.floor(logk.min()),np.ceil(logk.max())+dk,dk)
    Ei = np.zeros_like(logki)

    for i in range(logki.size):

        f = (logk>logki[i]-dk/2) & (logk<logki[i]+dk/2)

        if f.sum():
            Ei[i] = E[f].mean()
        else:
            Ei[i] = 0.

    ki = 10**logki
    fnnan = np.nonzero(Ei)
    Ei = Ei[fnnan]
    ki = ki[fnnan]

    return ki,Ei

