import os
import sys

from setuptools import setup


package_basename = 'CRStools'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='Rocher Antoine',
      author_email='antoine.rocher@epfl.ch',
      description='Tools for 4MOST-CRS work',
      license='GPLv3',
      url='https://github.com/4most-crs/4MOST_CRS_tools',
      install_requires=['matplotlib', 'numpy', 'healpy', 'astropy', 'scikit-learn'],
      #package_data={package_basename: ['*.mplstyle', 'data/*']},
      packages=[package_basename]
)