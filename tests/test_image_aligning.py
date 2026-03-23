"""
@author: kernke
"""


from numpy.testing import assert_allclose

from microscopy_data_analysis import image_aligning, image_processing


def test_plain_phase_correlation(
        central_pattern_img,
        shifted_pattern_img,
        expected_shift):
    res=image_processing.phase_correlation(central_pattern_img, shifted_pattern_img)
    assert_allclose(res[expected_shift[0],expected_shift[1]],1.)

def test_stack_shifting(stack):
    print(image_aligning.stack_shifting(stack))