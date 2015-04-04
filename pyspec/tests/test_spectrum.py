import numpy as np
from pyspec import spectrum as spec

def test_parseval(rtol=1.e-14, debug=False):
    """ Make sure 1D spectrum satisfied Parseval relation """
 
    N = np.array([10,11,21,500,871,1000,1001])
    dta = np.array([.25,1.,1.67,6.,20.])

    for dt in dta: 

        for n in N:
        
            phi = np.random.randn(n)
            spec_phi = spec.Spectrum(phi,dt)

            # var(P) from Fourier coefficients
            P_var_spec = spec_phi.var 

            # var(P) in physical space
            P_var_phys = phi.var()

            # relative error
            error = np.abs(P_var_phys - P_var_spec)/P_var_phys

            if debug:
                print "N = %i,  dt = %2.5f" %(n,dt)
                print "Variance in physical space: %5.16f" %P_var_phys
                print "Variance from spectrum: %5.16f" %P_var_spec
                print "error = %5.16f" %error
                print " "

            assert error<rtol, " *** Does not satisfy Parseval's relation "

if __name__ == "__main__":
    test_parseval(debug=True)
