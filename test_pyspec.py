import numpy as np
import matplotlib.pyplot as plt

from pyspec import spectrum as spec

n, L = 100, 2.
phi = np.random.randn(n)

phi_spec = spec.Spectrum(phi,L)
print phi.var()



uv_synthetic = np.load('synthetic_uv.npz')
up = uv_synthetic['up']

phi2 = np.random.randn(256,256)
spec2 = spec.TWODimensional_spec(up,1.,1.)

spec3 = spec2.spec
spec3 = np.ma.masked_array(spec3,np.sqrt(spec2.kappa2) > spec2.kk1.max())


