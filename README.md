# pyspec # 
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.31596.svg)](http://dx.doi.org/10.5281/zenodo.31596)
## A Pythonic package for spectral analysis ##


This is an on-going project in which I'm gathering codes for spectral analysis that I've been developing for my own research in oceanography. In particular, most of these codes were developed for a project on the horizontal wavenumber spectra in Drake Passage ([dp_spectra](https://github.com/crocha700/dp_spectra)). Contributions are very welcome.

## Dependencies ##
pyspec assumes you have basic Python packages for scientific applications (scipy, numpy, etc). I strongly encourage you to install the free [anaconda](https://store.continuum.io/cshop/anaconda/) distribution.

## Installation ##

This is a legit python package. You can install it

	python setup.py install

If you want to develop and contribute to the project, set up the development mode

	python setup.py develop

## Usage ##

Here are some simple examples 

* An IPython [notebook](http://nbviewer.ipython.org/github/crocha700/pyspec/blob/master/examples/example_1d_spec.ipynb) describing a single calculation of one-dimensional wavenumber spectra.

* An IPython [notebook](http://nbviewer.ipython.org/github/crocha700/dp_spectra/blob/master/adcp/buhler_etal_decomposition.ipynb) showcasing the decomposition of one-dimensional kinetic energy spectra into rotational and divergent components.

* An IPython [notebook](http://nbviewer.ipython.org/github/crocha700/pyspec/blob/master/examples/example_2d_spectra.ipynb) showing the basic usage of **pyspec** to compute 2D spectrum and its associated isotropic spectrum. This notebook also showcases the estimation of confidence limits.

## Funding ##
Part of this package was developed for a project funded by the NASA Ocean Surface Topography Science Team (NNX13AE44G) and the NSF Polar Program (PLR-1341431). The leading developer is currently supported by NSF (OCE 1357047).


