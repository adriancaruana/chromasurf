import numpy as np


CIEXYZ_TO_SRGB_MATRIX = np.array(
    (
        (3.24071, -1.53726, -0.498571),
        (-0.969258, 1.87599, 0.0415557),
        (0.0556352, -0.203996, 1.05707),
    )
)
SRGB_TO_CIEXYZ_MATRIX = np.array(
    (
        (0.412424, 0.357579, 0.180464),
        (0.212656, 0.715158, 0.0721856),
        (0.0193324, 0.119193, 0.950444),
    )
)
XN_YN_ZN_D65 = np.array((95.0489, 100, 108.8840))
YN_UNP_VNP_D65 = np.array((100, 0.2009, 0.4610))