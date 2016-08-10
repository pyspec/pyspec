import numpy as np
from numpy import pi,sinh,cosh
from scipy import integrate

try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

def diff_central(x, y):
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    f = (x2 - x1)/(x2 - x0)
    return (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)

def spec_helm_decomp(k,Cu,Cv,GM=False):

    """  it computes the Buhler et al  JFM 2014
            Helmholtz decomposition. That is,
            it splits the across-track/along-track
            KE spectra into rotational and divergent
            components.

    Inputs
    ==========
    - k: wavenumber
    - Cu: spectrum of across-track velocity
    - Cv: spectrum of along-track velocity

    Outputs
    ==========
    - Cpsi: rotational component of the KE spectrum
    - Cphi: divergent component of the KE spectrum

                                                  """
    dk = k[1]-k[0]
    s = np.log(k)

    Fphi = np.zeros_like(Cu)
    Fpsi = np.zeros_like(Cu)
    Cphi = np.zeros_like(Cu)
    Cpsi = np.zeros_like(Cu)

    # assume GM for decomposing into wave and vortex
    if GM:
        gm = np.load("/Users/crocha/Projects/dp_spectra/GM/gm_omega_star.npz")
        f2omg2 = gm['rgm']
        ks = gm['k']*1.e3

    for i in range(s.size-1):

        ds = np.diff(s[i:])

        sh = sinh(s[i]-s[i:])
        ch = cosh(s[i]-s[i:])

        # the function to integrate
        Fp = Cu[i:]*sh + Cv[i:]*ch
        Fs = Cv[i:]*sh + Cu[i:]*ch

        # integrate using Simpson's rule
        Fpsi[i] = integrate.simps(Fs,s[i:])
        Fphi[i] = integrate.simps(Fp,s[i:])

        # zero out unphysical values
        Fpsi[Fpsi < 0.] = 0.
        Fphi[Fphi < 0.] = 0.

    # compute rotational and divergent components
    Cpsi = Fpsi - Fphi + Cu
    Cphi = Fphi - Fpsi + Cv

    if GM:

        f2omg2i = np.interp(k,ks,f2omg2)

        Cv_w = f2omg2i*Fphi - Fpsi + Cv
        Cv_v = Cv - Cv_w
    
        kdkromg = diff_central(ks, f2omg2)
        kdkromg  = np.interp(k,ks[1:-1],kdkromg)

        dFphi =  diff_central(k, Fphi)
        #dFphi = np.gradient(Fphi,k)
        dFphi  = np.interp(k,k[1:-1],dFphi.real)
        E_w = Fphi - k*dFphi

        Cu_w = -k*kdkromg*Fphi + f2omg2i*(-Fpsi+Cv) + Fphi
        Cu_v = Cu - Cu_w

        Cb_w = E_w - (Cu_w + Cv_w)/2.

        return Cpsi,Cphi, Cu_w,Cv_w, Cu_v,Cv_v, E_w, Cb_w

    else:
        return Cpsi,Cphi



