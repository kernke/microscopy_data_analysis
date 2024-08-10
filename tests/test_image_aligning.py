# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:35:09 2024

@author: kernke
"""

import pytest

from numpy.testing import assert_allclose
from microscopic_data_analysis import image_aligning

def test_phase_correlation(
        central_pattern_img,
        shifted_pattern_img,
        expected_shift):
    res=image_aligning.phase_correlation(central_pattern_img, shifted_pattern_img)
    assert_allclose(res[expected_shift[0],expected_shift[1]],1.)

def test_stack_shifting(stack):
    print(image_aligning.stack_shifting(stack))