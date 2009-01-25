# Copyright 2009 Qiqi Wang
#
# This file is part of wanginterp.
#
# wanginterp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Interpolation and regression in multi-dimensional space

reference:
  Q Wang et al. A Multivariate Rational Interpolation
    Scheme with High Rates of Convergence.
    Submitted to Journal of Computational Physics.
"""


import copy
import sys
import pickle
import time

import numpy
import pylab
from numpy import zeros, ones, eye, kron, linalg, dot, exp, sqrt, diag, pi, \
                  asarray, sign

from _interp1d import _factorial, _solve_L, _solve_R, \
                      _lsqr_wang, _lsqr_golub




# ---------------------------------------------------------------------------- #




class InterpND(object):
  """
  Interpolation in multi-dimensional space.
  xv is the ``value data points'', i.e. the points
    where the function value is available and given
    by fxv; dfxv estimates the standard deviation of
    the error in the function values; the default of
    dfxv is 0 (fxv is exact).
  xg is the ``gradient data points'', i.e. points
    where the function gradient is available and given
    by fpxg; dfpxg estimates the standard devisiton of
    the error in the gradient values; the default of
    dfpxg is 0 (dfxg is exact).
  beta is the `magnitude' of the target function,
    can be automatically calculated.
  gamma is the `wave number' of the target function,
    can be automatically calculated.
    Combined with beta, it provides an estimate of
    the derivative growth: f^(k) = O(beta * gamma**k)
    Larger gamma yields more conservative, more robust
    and lower order interpolation.
  N is the order of the Taylor expansion, can be
    automatically calculated.  Smaller N yields lower
    order interpolation.  Numerical instability may
    occur when N is too large.
  l is the polynomial order.  The interpolant is
    forced to interpolate order l-1 polynomials
    exactly.  l=1 is the most robust, higher l makes
    a difference only when gamma is large, or when
    data is sparse and oscilatory if gamma is
    automatically calculated.
  verbose is the verbosity level.  0 is silent.
  Reference:
  * Q.Wang et al. A Rational Interpolation Scheme with
    Super-polynomial Rate of Convergence.
  """

  def __init__(self, xv, fxv, dfxv=None, xg=None, fpxg=None, dfpxg=None, \
               beta=None, gamma=None, N=None, l=1, verbose=1):
    """
    __init__(self, xv, fxv, dfxv=None, xg=None,
             fpxg=None, dfpxg=None, beta=None,
             gamma=None, N=None, l=1)
    Instantiation function, see class documentation
      for arguments.
    fxv must has same size as xv.
    dfxv must has same size as xv, or None for default
      (all 0).
    fpxg must be None if xg is None, or has same size
      as xg if xg is not None.
    dfpxg must be None if xg is None; if xg is not
      None it must has same size as xg, or None for
      default (all 0).
    When beta and gamma must be both None or both given.
      When they are None, their values are automatically
      calculated.  The calculation of gamma may take a
      long time if the number of datapoints is large.
    """
    pass
