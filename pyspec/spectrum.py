import numpy as np
from numpy import pi
from scipy.special import gammainc
from scipy import signal
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

        win =  np.hanning(self.n)
        win =  np.sqrt(self.n/(win**2).sum())*win

        self.phi *= win

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

        self.df = 1./(self.n*self.dt)

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

    def __init__(self,phi,d1,d2,detrend=True):

        self.phi = phi  # two dimensional real field
        self.d1 = d1
        self.d2 = d2
        self.n2,self.n1 = phi.shape
        self.L1 = d1*self.n1
        self.L2 = d2*self.n2

        if detrend:
            self.phi = signal.detrend(self.phi,axis=(-1),type='linear')
            self.phi = signal.detrend(self.phi,axis=(-2),type='linear')
        else:
            pass

        win1 =  np.hanning(self.n1)
        win1 =  np.sqrt(self.n1/(win1**2).sum())*win1
        win2 =  np.hanning(self.n2)
        win2 =  np.sqrt(self.n2/(win2**2).sum())*win2

        win = win1[np.newaxis,...]*win2[...,np.newaxis]

        self.phi *= win

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
        #self.calc_ispec()

        #self.ki,self.ispec =  calc_ispec(self.k1,self.k2,self.spec)

        self.spec =  np.fft.fftshift(self.spec,axes=0)

    def calc_freq(self):
        """ calculate array of spectral variable (frequency or
                wavenumber) in cycles per unit of L """

        # wavenumber one (equals to dk1 and dk2)
        self.dk1 = 1./self.L1
        self.dk2 = 1./self.L2

        # wavenumber grids
        self.k2 = self.dk2*np.append( np.arange(0.,self.n2/2), \
                  np.arange(-self.n2/2,0.) )
        self.k1 = self.dk1*np.arange(0.,self.n1/2+1)

        self.kk1,self.kk2 = np.meshgrid(self.k1,self.k2)

        self.kk1 = np.fft.fftshift(self.kk1,axes=0)
        self.kk2 = np.fft.fftshift(self.kk2,axes=0)
        self.kappa2 = self.kk1**2 + self.kk2**2
        self.kappa = np.sqrt(self.kappa2)


    def calc_spectrum(self):
        """ calculates the spectrum """
        self.phih = np.fft.rfft2(self.phi)
        self.spec = 2.*(self.phih*self.phih.conj()).real/ (self.dk1*self.dk2)\
                / (self.n1*self.n2)**2

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


class THREEDimensional_spec(object):
    """ A class that represent a three dimensional spectrum
            for real signals """

    def __init__(self,phi,d1,d2,d3):

        self.phi = phi
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.n2,self.n1, self.n3 = phi.shape
        self.L1 = d1*self.n1
        self.L2 = d2*self.n2
        self.L3 = d3*self.n3

        win1 =  np.hanning(self.n1)
        win1 =  (self.n1/(win1**2).sum())*win1
        win2 =  np.hanning(self.n2)
        win2 =  (self.n2/(win2**2).sum())*win2
        win3 =  np.hanning(self.n3)
        win3 =  (self.n3/(win3**2).sum())*win3

        win = win1[np.newaxis]*win2[...,np.newaxis]
        win = win[...,np.newaxis]*win3[np.newaxis,np.newaxis]

        self.phi = self.phi*win

        # calculate frequencies
        self.calc_freq()

        # calculate spectrum
        self.calc_spectrum()

        # the isotropic spectrum
        #self.ki,self.ispec =  calc_ispec(self.k1,self.k2,self.spec,ndim=3)


    def calc_freq(self):
        """ calculate array of spectral variable (frequency or
                wavenumber) in cycles per unit of L """

        # wavenumber one (equals to dk1 and dk2)
        self.dk1 = 1./self.L1
        self.dk2 = 1./self.L2
        self.dk3 = 1./self.L3

        # wavenumber grids
        #self.k1 = self.dk1*np.append( np.arange(0.,self.n1/2), \
        #          np.arange(-self.n1/2+1,0.) )
        self.k1 = np.fft.fftfreq(self.n1, d=self.d1)
        self.k2 = np.fft.fftfreq(self.n2, d=self.d2)
        #self.k2 = self.dk2*np.append( np.arange(0.,self.n2/2), \
        #      np.arange(-self.n2/2,0.) )
        self.k3 = self.dk3*np.arange(0.,self.n3/2+1)

        self.kk1,self.kk2,self.kk3 = np.meshgrid(self.k1,self.k2,self.k3)

        self.kk1 = np.fft.fftshift(self.kk1,axes=0)
        self.kk2 = np.fft.fftshift(self.kk2,axes=0)
        self.kk3 = np.fft.fftshift(self.kk3,axes=0)

        self.kappa2 = self.kk1**2 + self.kk2**2 + self.kk3**2
        self.kappa = np.sqrt(self.kappa2)

    def calc_spectrum(self):
        """ calculates the spectrum """
        self.phih = np.fft.rfftn(self.phi,axes=(0,1,2))
        self.spec = 2.*(self.phih*self.phih.conj()).real/(self.dk1*self.dk2*self.dk3)\
                / (self.n1*self.n2*self.n3)**2

# utilities
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
        lower (El) and upper (Eu) bounds on E """

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


# for llc output only; this is temporary
def spec_est(A,d,axis=2,window=True,detrend=True, prewhiten=False):

    l1,l2,l3,l4 = A.shape

    if axis==2:
        l = l3
        if prewhiten:
            if l3>1:
                _,A,_ = np.gradient(A,d,d,1.)
            else:
                _,A = np.gradient(A.squeeze(),d,d)
                A = A[...,np.newaxis]
        if detrend:
            A = signal.detrend(A,axis=axis,type='linear')
        if window:
            win = np.hanning(l)
            win = (l/(win**2).sum())*win
            win = win[np.newaxis,np.newaxis,:,np.newaxis]
        else:
            win = np.ones(l)[np.newaxis,np.newaxis,:,np.newaxis]
    elif axis==1:
        l = l2
        if prewhiten:
            if l3 >1:
                A,_,_ = np.gradient(A,d,d,1.)
            else:
                A,_ = np.gradient(A.squeeze(),d,d)
                A = A[...,np.newaxis]
        if detrend:
            A = signal.detrend(A,axis=1,type='linear')
        if window:
            win = np.hanning(l)
            win = (l/(win**2).sum())*win
            win = win[np.newaxis,...,np.newaxis,np.newaxis]
        else:
            win = np.ones(l)[np.newaxis,...,np.newaxis,np.newaxis]


    df = 1./(d*l)
    f = np.arange(0,l/2+1)*df

    Ahat = np.fft.rfft(win*A,axis=axis)
    Aabs = 2 * (Ahat*Ahat.conjugate()) / l

    if prewhiten:

        if axis==1:
            fd = 2*np.pi*f[np.newaxis,:, np.newaxis]
        else:
            fd = 2*np.pi*f[...,np.newaxis,np.newaxis]

        Aabs = Aabs/(fd**2)
        Aabs[0,0] = 0.

    return Aabs,f

# for llc output only; this is temporary
#def spec_est(A,d,axis=1,window=True,detrend=True, prewhiten=False):
#
#    l1,l2,l3 = A.shape
#
#    if axis==1:
#        l = l2
#        if prewhiten:
#            if l3>1:
#                _,A,_ = np.gradient(A,d,d,1.)
#            else:
#                _,A = np.gradient(A.squeeze(),d,d)
#                A = A[...,np.newaxis]
#        if detrend:
#            A = signal.detrend(A,axis=1,type='linear')
#        if window:
#            win = np.hanning(l)
#            win = (l/(win**2).sum())*win
#            win = win[np.newaxis,:,np.newaxis]
#        else:
#            win = np.ones(l)[np.newaxis,:,np.newaxis]
#    else:
#        l = l1
#        if prewhiten:
#            if l3 >1:
#                A,_,_ = np.gradient(A,d,d,1.)
#            else:
#                A,_ = np.gradient(A.squeeze(),d,d)
#                A = A[...,np.newaxis]
#        if detrend:
#            A = signal.detrend(A,axis=0,type='linear')
#        if window:
#            win = np.hanning(l)
#            win = (l/(win**2).sum())*win
#            win = win[...,np.newaxis,np.newaxis]
#        else:
#            win = np.ones(l)[...,np.newaxis,np.newaxis]
#
#    df = 1./(d*l)
#    f = np.arange(0,l/2+1)*df
#
#    Ahat = np.fft.rfft(win*A,axis=axis)
#    Aabs = 2 * (Ahat*Ahat.conjugate()) / l
#
#    if prewhiten:
#
#        if axis==1:
#            fd = 2*np.pi*f[np.newaxis,:, np.newaxis]
#        else:
#            fd = 2*np.pi*f[...,np.newaxis,np.newaxis]
#
#
#        Aabs = Aabs/(fd**2)
#        Aabs[0,0] = 0.
#
#    return Aabs,f

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

def calc_ispec(k,l,E,ndim=2):
    """ Calculates the azimuthally-averaged spectrum

        Parameters
        ===========
        - E is the two-dimensional spectrum
        - k is the wavenumber is the x-direction
        - l is the wavenumber in the y-direction

        Output
        ==========
        - kr: the radial wavenumber
        - Er: the azimuthally-averaged spectrum """

    dk = np.abs(k[2]-k[1])
    dl = np.abs(l[2]-l[1])

    k, l = np.meshgrid(k,l)

    wv = np.sqrt(k**2+l**2)

    if k.max()>l.max():
        kmax = l.max()
    else:
        kmax = k.max()

    if ndim==3:
        nl, nk, nomg = E.shape
    elif ndim==2:
        nomg = 1

    dkr = np.sqrt(dk**2 + dl**2)
    kr =  np.arange(dkr/2.,kmax+dkr/2.,dkr)
    Er = np.zeros((kr.size,nomg))


    for i in range(kr.size):

        fkr =  (wv>=kr[i]-dkr/2) & (wv<=kr[i]+dkr/2)
        dth = np.pi / (fkr.sum()-1)
        if ndim==2:
            Er[i] = (E[fkr]*(wv[fkr]*dth)).sum()
        elif ndim==3:
            Er[i] = (E[fkr]*(wv[fkr]*dth)).sum(axis=(0,1))


    return kr, Er.squeeze()

def spectral_slope(k,E,kmin,kmax,stdE):
    ''' compute spectral slope in log space in
        a wavenumber subrange [kmin,kmax],
        m: spectral slope; mm: uncertainty'''

    fr = np.where((k>=kmin)&(k<=kmax))

    ki = np.matrix((np.log10(k[fr]))).T
    Ei = np.matrix(np.log10(np.real(E[fr]))).T
    dd = np.matrix(np.eye(ki.size)*((np.abs(np.log10(stdE)))**2))

    G = np.matrix(np.append(np.ones((ki.size,1)),ki,axis=1))
    Gg = ((G.T*G).I)*G.T
    m = Gg*Ei
    mm = np.sqrt(np.array(Gg*dd*Gg.T)[1,1])
    yfit = np.array(G*m)
    m = np.array(m)[1]

    return m, mm
