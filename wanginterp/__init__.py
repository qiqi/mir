"""
wanginterp verson 1.0.0
Qiqi Wang  January 2009

Interpolation and regression in one-dimensional space
Interpolation and regression in two-dimensional space
Interpolation and regression in multi-dimensional space

reference:
  Q Wang et al. A Rational Interpolation Scheme with
    Super-Polynomial Rate of Convergence.
    In CTR Anual Research Briefs 2008.
    Submitted to SIAM Journal of Numerical Analysis.

  Q Wang et al. A Multivariate Rational Interpolation
    Scheme with High Rate of Convergence.
    In preparation.

  Q Wang et al. A High Order Meshless Method for
    Partial Differential Equations in Complex Geometry.
    In preparation.
"""


from _interp1d import Interp1D
from _interp1dvg import Interp1DVG
from _interp2d import Interp2D
from _interpnd import InterpND
