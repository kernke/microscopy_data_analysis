# -*- coding: utf-8 -*-
"""
@author: kernke
"""
from setuptools import setup
from setuptools import find_packages


setup(name='microscopic_data_analysis',
      version='0.1',
      description='microscopic_data_analysis',
      url='https://github.com/kernke/microscopic_data_analysis',
      author='Kernke',
      author_email='r.s.kernke@gmail.com',
      license='AGPL-3.0',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
            'numpy',
            'opencv-python',
            'scipy',
            'numba',
            'h5py',
            'scikit-image',
            'imageio',
            'sphinx-rtd-theme',
            'ncempy'
               ])
