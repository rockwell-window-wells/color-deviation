"""
CIE L*a*b* <-> CIE XYZ conversions, D65 2-degree standard observer white point.
"""

# CIE D65, 2-degree standard observer
WHITE_D65 = (95.047, 100.000, 108.883)

_DELTA = 6 / 29


def _f_inv(t):
    if t > _DELTA:
        return t ** 3
    return 3 * _DELTA ** 2 * (t - 4 / 29)


def _f(t):
    if t > _DELTA ** 3:
        return t ** (1 / 3)
    return t / (3 * _DELTA ** 2) + 4 / 29


def lab_to_xyz(L, a, b, white=WHITE_D65):
    Xn, Yn, Zn = white
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    return Xn * _f_inv(fx), Yn * _f_inv(fy), Zn * _f_inv(fz)


def xyz_to_lab(X, Y, Z, white=WHITE_D65):
    Xn, Yn, Zn = white
    fx = _f(X / Xn)
    fy = _f(Y / Yn)
    fz = _f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b
