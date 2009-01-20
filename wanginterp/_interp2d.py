"""
Interpolation and regression in two-dimensional space

reference:
  Q Wang et al. A Multivariate Rational Interpolation
    Scheme with High Rates of Convergence.
    Submitted to Journal of Computational Physics.

Qiqi Wang  January 2009
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




class Interp2D(object):
  """
  Interpolation in 2D.
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
    
    assert verbose == 0 or verbose == 1 or verbose == 2
    self.verbose = verbose

    # verify and save value data points
    assert xv.ndim == 2 and fxv.ndim == 1
    assert xv.shape[1] == 2 and xv.shape[0] == fxv.shape[0]
    self.xv = copy.copy(xv)
    self.fxv = copy.copy(fxv)
    if dfxv is None:
      self.dfxv = zeros(fxv.shape)
    else:
      assert dfxv.shape == fxv.shape
      self.dfxv = copy.copy(dfxv)

    # verify and save gradient data points
    if xg is None:
      assert fpxg is None and dfpxg is None
      self.xg = zeros([0,2])
      self.fpxg = zeros([0,2])
      self.dfpxg = zeros(0)
    else:
      assert xg.ndim == 2 and xg.shape[1] == 2
      assert fpxg is not None and fpxg.shape == xg.shape
      self.xg = copy.copy(xg)
      self.fpxg = copy.copy(fpxg)
      if dfpxg is None:
        self.dfpxg = zeros(xg.shape[0])
      else:
        assert dfpxg.ndim == 1 and dfpxg.shape[0] == xg.shape[0]
        self.dfpxg = copy.copy(dfpxg)

    # check and automatically calculate N
    self.nv = self.xv.shape[0]
    self.ng = self.xg.shape[0]
    self.n = self.nv + self.ng * 2
    if N is None:
      self.N = min(self.n, 200)
    else:
      self.N = N

    assert int(l) == l
    self.l = int(l)

    # automatically calculate beta and gamma
    if beta is None:
      assert gamma is None
      self.beta = self.calc_beta()
      self.gamma = self.calc_gamma()
    else:
      self.beta = float(beta)
      self.gamma = float(gamma)




