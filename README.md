[![Python package](https://github.com/kernke/microscopy_data_analysis/actions/workflows/python-package.yml/badge.svg)](https://github.com/kernke/microscopy_data_analysis/actions/workflows/python-package.yml)

# microscopy_data_analysis

Microscopy data most of the times relates to images (i.e., 2D-data). However, sometimes spectral information represents just one-dimensional data or it extends an image by a spectrum at every pixel to a three-dimensional dataset. In a similar way the focus of this package lays on 2D-datasets, but also extends from time to time.

Main features are:

 stitching images on a grid
 
 stack aligning
 
 line detection
 
# Installation
After changing into the directory containing setup.py , install via 

    pip install -r /docs/requirements.txt
    pip install .  


Documentation: https://microscopy-data-analysis.readthedocs.io

To make sure everything works fine, tests can be executed by typing

    pytest
