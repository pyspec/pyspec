from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyspec',
      version='0.1dev1',
      description='A pythonian package for spectral\
              analysis',
      url='http://github.com/crocha700/pyspec',
      author='Cesar B Rocha',
      author_email='crocha@ucsd.edu',
      license='MIT',
      packages=['pyspec'],
      install_requires=[
          'numpy',
      ],
      test_suite = 'nose.collector',
      zip_safe=False)
