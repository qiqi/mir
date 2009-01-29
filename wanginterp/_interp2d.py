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
Interpolation and regression in two-dimensional space

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




def _binomial(m, n):
  assert n >= 0 and m >= n
  b = 1
  for k in range(1,n+1):
    b  = b * (m-n+k) / k
  return b


def _binomial_inv(l, n):
  "Calculate the largest b such that _binomial(b,n) <= l."
  m, b = n, 1
  while b <= l:
    m += 1
    b = b * m / (m-n)
  return m - 1


def _order_set_2d(N):
  if N == 1:
    return [[0,1], [1,0]]
  else:
    set = _order_set_2d(N-1)
    set.extend([i,N-i] for i in range(N+1))
    return set



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
               beta=None, gamma=None, N=None, l=1, verbose=1, \
               safety_factor=1.0):
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

    # check and save safety factor
    assert safety_factor > 0.0
    self.safety_factor = float(safety_factor)

    # check and automatically calculate N
    self.nv = self.xv.shape[0]
    self.ng = self.xg.shape[0]
    self.n = self.nv + self.ng * 2
    if N is None:
      self.N = min(_binomial_inv(self.n,2), 20)
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



  def interp_matrices(self, x, beta=None, gamma=None):
    """
    Calculate the matrices X, E and C for interpolation
    scheme x is the point where the interpolation is
    evaluated.
    """
    assert x.dtype == float
    N, n, l, nv, ng = self.N, self.n, self.l, self.nv, self.ng
    if beta is None:
      beta = self.beta
    if gamma is None:
      gamma = self.gamma

    # construct the order set
    order_set = _order_set_2d(N+1)
    M1 = _binomial(N+2,2) - 1
    M2 = _binomial(N+3,2) - 1
    assert len(order_set) == M2

    # construct X = [Xv,Xg]
    X = zeros([M1, n], dtype=float)
    for i, kappa in enumerate(order_set[:M1]):
      X[i,:nv] = gamma**sum(kappa) / _factorial(kappa) * \
                 ((self.xv - x)**kappa).prod(1)
      if nv > 0:
        if kappa[0] > 0:
          kappa_p = [kappa[0]-1, kappa[1]]
          X[i,nv:nv+ng] = gamma**sum(kappa) / _factorial(kappa_p) * \
                          ((self.xg - x)**kappa_p).prod(1)
        if kappa[1] > 0:
          kappa_p = [kappa[0], kappa[1]-1]
          X[i,nv+ng:] = gamma**sum(kappa) / _factorial(kappa_p) * \
                        ((self.xg - x)**kappa_p).prod(1)
    X *= beta

    # construct diagonal G matrix for the Lagrange residual
    Er2 = zeros(n)
    for kappa in order_set[M1:]:
      Eri = gamma**sum(kappa) / _factorial(kappa) * \
            ((self.xv - x)**kappa).prod(1)
      Er2[:nv] += Eri**2
      if ng > 0:
        if kappa[0] > 0:
          kappa_p = [kappa[0]-1, kappa[1]]
          Eri = gamma**sum(kappa) / _factorial(kappa_p) * \
                ((self.vg - x)**kappa_p).prod(1)
          Er2[nv:nv+ng] += Eri**2
        if kappa[1] > 0:
          kappa_p = [kappa[0], kappa[1]-1]
          Eri = gamma**sum(kappa) / _factorial(kappa_p) * \
               ((self.vg - x)**kappa_p).prod(1)
          Er2[nv+ng:] += Eri**2
    Er2 *= beta**2

    # construct diagonal H matrix for measurement errors
    Ee = zeros(n)
    Ee[:nv] = self.dfxv
    if ng > 0:
      Ee[nv:nv+ng] = self.dfpxg
      Ee[nv+ng:] = self.dfpxg

    # construct E
    E = sqrt(Er2 + Ee**2)

    # construct C
    M = _binomial(l+1,2) - 1
    assert M <= len(order_set)
    C = zeros([M+1, n])
    C[0,:nv] = 1.0
    C[0,nv:] = 0.0
    for i, kappa in enumerate(order_set[:M]):
      C[i+1,:nv] = (self.xv - x)**kappa
      if ng > 0:
        kappa_p = [kappa[0]-1, kappa[1]]
        C[i+1,nv:nv+ng] = kappa[0] * (self.xv - x)**kappa_p
        kappa_p = [kappa[0], kappa[1]-1]
        C[i+1,nv:nv+ng] = kappa[1] * (self.xv - x)**kappa_p
    return X, E, C



  def interp_coef(self, x, beta=None, gamma=None):
    """
    Calculate interpolation coefficients in two-dimensional
      spacee.
    x is the point where the interpolation is evaluated.
    beta is the `magnitude' of the target function.
    gamma is the `wave number' of the target function.
      combined with beta, it provides an estimate of the derivative growth:
        f^(k) = O(beta * gamma**k)
      larger gamma = more conservative and lower order interpolation.
    return values:
      av, ag, er2 = interp.interp_2d_coef(x)
    av and ag are the interpolation coefficients.
    er2 is the expected squared residual.
    """
    x = numpy.asarray(x, dtype=float)
    # calculate interpolation coefficients at x
    if abs(self.xv - x).sum(1).min() < 1.0E-12:   # exactly matches a data point
      imatch = abs(self.xv - x).sum(1).argmin()
      if abs(self.dfxv[imatch]).sum() == 0.0:
        # and function value on that point is exact
        av = numpy.array([0]*imatch+[1]+[0]*(self.nv-imatch-1))
        ag = numpy.zeros(self.xg.shape)
        return av, ag, 0.0
    # construct matrices
    X, E, C = self.interp_matrices(x, beta, gamma)
    # assemble the diagonal of matrix A, and sort by its diagonal
    diagA = (X**2).sum(0) + E**2
    isort = sorted(range(self.n), key=diagA.__getitem__)
    irevt = sorted(range(self.n), key=isort.__getitem__)
    # permute columns of X and diagonal of G and H
    X = X[:,isort]
    E = E[isort]
    C = C[:,isort]
    # solve least square
    if C.shape[0] == 1:
      a = _lsqr_wang(X, E, C[0,:])
    else:
      b = zeros(X.shape[0] + X.shape[1])
      d = zeros(C.shape[0])
      d[0] = 1.0
      a = _lsqr_golub(X, E, b, C, d)
    # reverse sorting permutation to get a and b
    arevt = a[irevt]
    av = arevt[:self.nv]
    ag = arevt[self.nv:].reshape([self.ng,2])
    # compute the expeted squared residual
    finite = (a != 0)
    Xa = dot(X[:,finite], a[finite])
    Ea = (E*a)[finite]
    er2 = (Xa**2).sum() + (Ea**2).sum()
    return av, ag, er2



  def calc_beta(self):
    """
    Estimate the `magnitude' parameter beta from data points.
    """
    if self.dfxv.max() == 0:  # beta does not matter in this case.
      return 1.0
    else:
      assert self.fxv.ndim == 1 and self.fxv.shape == self.dfxv.shape
      f_bar = self.fxv.mean()
      ratio = (self.dfxv**2).sum() / ((self.fxv - f_bar)**2).sum() * \
              float(self.nv-1) / float(self.nv)
      beta = sqrt(((self.fxv - f_bar)**2).sum() / (self.nv-1) * exp(-ratio))
      return beta



  def _calc_gamma_bounds(self):
    """
    Calculate lower and upper bounds for gamma based
      on the distribution of grid points.
    Returns (gamma_min, gamma_max) pair.
    """
    delta_min, delta_max = numpy.inf, 0.0
    for xi in list(self.xv) + list(self.xg):
      d2xv = ((xi - self.xv)**2).sum(1)
      if self.ng > 0:
        d2xg = ((xi - self.xg)**2).sum(1)
        delta_max = max(delta_max, max(d2xv.max(), d2xg.max()))
      else:
        delta_max = max(delta_max, d2xv.max())
    for i, xi in enumerate(self.xv):
      if i > 0:
        d2x = ((xi - self.xv[:i,:])**2).sum(1)
        delta_min = min(delta_min, d2x.min())
    for i, xi in enumerate(self.xg):
      if i > 0:
        d2x = ((xi - self.xg[:i,:])**2).sum(1)
        delta_min = min(delta_min, d2x.min())
    delta_min, delta_max = sqrt(delta_min), sqrt(delta_max)
    assert delta_max > delta_min
    gamma_min = 1. / delta_max
    gamma_max = pi / delta_min
    return gamma_min, gamma_max



  def _calc_res_ratio(self, iv, beta, gamma, safety_factor=None):
    """
    A utility function used by calc_gamma, calculates
      the ratio of real residual to the estimated
      residual at the iv'th value data point, which
      is used to make decision in the bisection process
      for gamma at the iv'th data point.
    """
    if safety_factor is None:
      safety_factor = self.safety_factor
    base = range(iv) + range(iv+1,self.nv)
    subinterp = Interp2D(self.xv[base], self.fxv[base], self.dfxv[base], \
                         self.xg, self.fpxg, self.dfpxg, beta, gamma, \
                         self.N, self.l, self.verbose)
    av, ag, er2 = subinterp.interp_coef(self.xv[iv])
    resid = dot(av,self.fxv[base]) + dot(ag.flat,self.fpxg.flat) - self.fxv[iv]
    return resid**2 / (er2 + self.dfxv[iv]**2) * safety_factor



  def calc_gamma(self):
    """
    Estimate the `wave number' parameter gamma from
      data points.  This function prints stuff when
      self.verbose > 1.
    """
    assert isinstance(self.beta, float)
    # logorithmic bisection for gamma
    gamma_min, gamma_max = self._calc_gamma_bounds()
    while gamma_max / gamma_min > 1.1:
      if self.verbose > 1:
        print '    bisecting [', gamma_min, ',', gamma_max, '] for gamma...'
      gamma_mid = sqrt(gamma_max * gamma_min)
      res_ratio = 0.0
      for i in range(self.nv):
        res_ratio += self._calc_res_ratio(i, self.beta, gamma_mid)
      res_ratio /= self.nv
      if res_ratio < 1.0:
        gamma_max = gamma_mid
      else:
        gamma_min = gamma_mid
    # final selected gamma
    gamma_mid = sqrt(gamma_max * gamma_min)
    if self.verbose > 1:
      print '    using gamma = ', gamma_mid
    return gamma_mid



  def interp(self, x, compute_df=False):
    """
    Interpolation in 2D.
    x is the point (size 2 array) or points
      (a list of size 2 arrays or an shape(n,2) array)
      where the interpolation is evaluated.
    compute_df indicates whether an estimated standard
      deviation of the error in the interpolation
      approximation is also returned.
    Usage:
      fx = interp(x, compute_df=False)";
      fx, df = interp(x, compute_df=True)"
    """
    # evaluate interpolant value at a single point
    x = numpy.array(x)
    if x.shape == (2,):
      av, ag, er2 = self.interp_coef(x)
      fx = dot(av, self.fxv) + dot(ag.flat, self.fpxg.flat)
      dfx = sqrt(er2)
      if compute_df:
        return fx, dfx
      else:
        return fx
    # at multiple points
    else:
      fx, dfx = [], []
      for xi in x:
        av, ag, er2 = self.interp_coef(xi)
        fx.append(dot(av, self.fxv) + dot(ag.flat, self.fpxg.flat))
        dfx.append(sqrt(er2))
      if compute_df:
        return numpy.asarray(fx), numpy.asarray(dfx)
      else:
        return numpy.asarray(fx)



