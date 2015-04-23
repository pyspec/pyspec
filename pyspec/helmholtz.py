import numpy as np
from numpy import pi,sinh,cosh
from scipy import integrate

try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

class BCFDecomposition(object):
      """ A class that represents the Buhler et al  JFM 2014 
            Helmholtz decomposition """
      def __init__(self,k,Cu,Cv):

        self.k = k
        self.Cu = Cu
        self.Cv = Cv
        self.dk = k[1]-k[0]
       
        s = np.log(self.k)
        
        Fphi = np.zeros_like(self.Cu)
        Fpsi = np.zeros_like(self.Cu)
        self.Cphi = np.zeros_like(self.Cu)
        self.Cpsi = np.zeros_like(self.Cu)

        for i in range(s.size-1):

            ds = np.diff(s[i:])

            sh = sinh(s[i]-s[i:])
            ch = cosh(s[i]-s[i:])

            cu = self.Cu[i:]
            cv = self.Cv[i:]

            # the function to integrate
            Fp = cu*sh + cv*ch
            Fs = cv*sh + cu*ch

            # integrate using simpsons rule
            Fpsi[i] = integrate.simps(Fs,s[i:])
            Fphi[i] = integrate.simps(Fp,s[i:])

        # zero out unphysical values
        Fpsi[Fpsi < 0.] = 0.
        Fphi[Fphi < 0.] = 0.

        # now compute rotational and divergent components
        self.Cpsi = Fpsi - self.k*np.gradient(Fpsi,self.dk)
        self.Cphi = Fphi - self.k*np.gradient(Fphi,self.dk)


        
        

