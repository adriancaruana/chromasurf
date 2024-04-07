import abc
import dataclasses
from typing import ClassVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from color import (
    ColorSpace,
    Color,
    CIEXYZColor,
    SRGBColor,
    CIELABColor,
    CIELUVColor,
    HSLColor,
)
from constants import (
    CIEXYZ_TO_SRGB_MATRIX,
    SRGB_TO_CIEXYZ_MATRIX,
    XN_YN_ZN_D65,
    YN_UNP_VNP_D65,
)


@dataclasses.dataclass(frozen=True)
class _Transformer(abc.ABC):
    space_to: ColorSpace = dataclasses.field(init=False)
    space_from: ColorSpace = dataclasses.field(init=False)

    @property
    def edge(self):
        return (self.space_from, self.space_to)

    @abc.abstractmethod
    def fwd(self, c: Color):
        pass

    @abc.abstractmethod
    def bwd(self, c: Color):
        pass


@dataclasses.dataclass(frozen=True)
class _CIEXYZ2SRGB(_Transformer):
    space_to: ClassVar[ColorSpace] = ColorSpace.SRGB
    space_from: ClassVar[ColorSpace] = ColorSpace.CIEXYZ
    _fwd_matrix: ClassVar[NDArray] = CIEXYZ_TO_SRGB_MATRIX
    _bwd_matrix: ClassVar[NDArray] = SRGB_TO_CIEXYZ_MATRIX

    def fwd(self, c: CIEXYZColor) -> SRGBColor:
        xyz = np.array(c.values)
        rgb_linear = np.dot(self._fwd_matrix, xyz)
        rgb = np.where(
            rgb_linear <= 0.0031308, 
            12.92 * rgb_linear, 
            1.055 * np.power(rgb_linear, 1/2.4) - 0.055
        )
        return SRGBColor(rgb)

    def bwd(self, c: SRGBColor) -> CIEXYZColor:
        rgb = np.array(c.values)
        rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
        xyz = np.dot(self._bwd_matrix, rgb_linear)
        return CIEXYZColor(xyz)
    

@dataclasses.dataclass(frozen=True)
class _CIEXYZ2CIELAB(_Transformer):
    space_to: ClassVar[ColorSpace] = ColorSpace.CIELAB
    space_from: ClassVar[ColorSpace] = ColorSpace.CIEXYZ
    # Assuming D65 illuminant as the reference white
    _XnYnZn: ClassVar[NDArray] = XN_YN_ZN_D65

    def _f(self, t: float) -> float:
        delta = 6/29
        if t > delta ** 3:
            return np.cbrt(t)
        else:
            return t / (3 * delta ** 2) + 4/29

    def _f_inv(self, t: float) -> float:
        delta = 6/29
        if t > delta:
            return t ** 3
        else:
            return 3 * delta ** 2 * (t - 4/29)

    def fwd(self, c: CIEXYZColor) -> CIELABColor:
        Xr, Yr, Zr = c.values / self._XnYnZn
        L = 116 * self._f(Yr) - 16
        a = 500 * (self._f(Xr) - self._f(Yr))
        b = 200 * (self._f(Yr) - self._f(Zr))
        return CIELABColor((L, a, b))

    def bwd(self, c: CIELABColor) -> CIEXYZColor:
        L, a, b = c.values
        fy = (L + 16) / 116
        fx = fy + a / 500
        fz = fy - b / 200
        Xn, Yn, Zn = self._XnYnZn
        X = self._f_inv(fx) * Xn
        Y = self._f_inv(fy) * Yn
        Z = self._f_inv(fz) * Zn
        return CIEXYZColor((X, Y, Z))


@dataclasses.dataclass(frozen=True)
class _CIEXYZ2CIELUV(_Transformer):
    space_to: ClassVar[ColorSpace] = ColorSpace.CIELUV
    space_from: ClassVar[ColorSpace] = ColorSpace.CIEXYZ
    _YnUnpVnp: ClassVar[float] = YN_UNP_VNP_D65

    def _uv_prime(self, X: float, Y: float, Z: float) -> tuple:
        denom = X + 15 * Y + 3 * Z
        u_prime = (4 * X) / denom if denom != 0 else 0
        v_prime = (9 * Y) / denom if denom != 0 else 0
        return u_prime, v_prime

    def fwd(self, c: CIEXYZColor) -> CIELUVColor:
        Yn, un_prime, vn_prime = self._YnUnpVnp
        X, Y, Z = c.values
        Yr = Y / Yn
        u_prime, v_prime = self._uv_prime(X, Y, Z)
        if Yr <= (6/29) ** 3:
            L_star = (29/3) ** 3 * Yr
        else:
            L_star = 116 * Yr ** (1/3) - 16
        u_star = 13 * L_star * (u_prime - un_prime)
        v_star = 13 * L_star * (v_prime - vn_prime)
        return CIELUVColor((L_star, u_star, v_star))

    def bwd(self, c: CIELUVColor) -> CIEXYZColor:
        Yn, un_prime, vn_prime = self._YnUnpVnp
        L_star, u_star, v_star = c.values
        if L_star <= 8:
            Y = Yn * L_star * ((3/29) ** 3)
        else:
            Y = Yn * (((L_star + 16) / 116) ** 3)
        u_prime = u_star / (13 * L_star) + un_prime if L_star != 0 else un_prime
        v_prime = v_star / (13 * L_star) + vn_prime if L_star != 0 else vn_prime
        X = Y * (9 * u_prime) / (4 * v_prime)
        Z = Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)
        return CIEXYZColor((X, Y, Z))
    

@dataclasses.dataclass(frozen=True)
class _SRGB2HSL(_Transformer):
    space_to: ClassVar[ColorSpace] = ColorSpace.HSL
    space_from: ClassVar[ColorSpace] = ColorSpace.SRGB

    def fwd(self, c: SRGBColor) -> HSLColor:
        r, g, b = c.values
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        L = (max_val + min_val) / 2

        if max_val == min_val:
            H = S = 0
        else:
            diff = max_val - min_val
            S = diff / (2 - max_val - min_val) if L > 0.5 else diff / (max_val + min_val)

            if max_val == r:
                H = (g - b) / diff + (6 if g < b else 0)
            elif max_val == g:
                H = (b - r) / diff + 2
            elif max_val == b:
                H = (r - g) / diff + 4
            H /= 6

        return HSLColor((H, S, L))

    def bwd(self, c: HSLColor) -> SRGBColor:
        def hue2rgb(p, q, t):
            t = t if t < 1 else t - 1 if t > 1 else t + 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        H, S, L = c.values
        if S == 0:
            r = g = b = L
        else:
            q = L * (1 + S) if L < 0.5 else L + S - L * S
            p = 2 * L - q
            r = hue2rgb(p, q, H + 1/3)
            g = hue2rgb(p, q, H)
            b = hue2rgb(p, q, H - 1/3)

        return SRGBColor((r, g, b))


TRANSFORMERS = [
    _CIEXYZ2SRGB,
    _CIEXYZ2CIELAB,
    _CIEXYZ2CIELUV,
    _SRGB2HSL,
]
TRANSFORMS = {
    tr().edge: tr
    for tr in TRANSFORMERS
}
TREE = nx.DiGraph(list(TRANSFORMS.keys()))

def transforms_and_ops(c: Color, to: ColorSpace | str) -> list[tuple[_Transformer, str]]:
    node_list = nx.shortest_path(TREE.to_undirected(), source=c._space, target=to)
    transforms = []
    ops = []
    for edge in zip(node_list[:-1], node_list[1:]):
        if edge in TRANSFORMS:
            transforms.append(TRANSFORMS[edge])
            ops.append("fwd")
        elif edge[::-1] in TRANSFORMS:
            transforms.append(TRANSFORMS[edge[::-1]])
            ops.append("bwd")
        else:
            raise ValueError(f"Edge {edge} not found in transform graph.")
    return transforms, ops


def transform(c: Color, to: ColorSpace | str) -> Color:
    transforms, ops = transforms_and_ops(c, to)
    for tr, op in zip(transforms, ops):
        print("Transforming", c, "using", tr, "with", op)
        c = tr().__getattribute__(op)(c)
    return c

