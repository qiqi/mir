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
  with variable gamma.

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
                  asarray, sign, log, exp

from _interp2d import Interp2D




# ---------------------------------------------------------------------------- #




class Interp2DVG(Interp2D):
  """
  Variable gamma version of Interp2D.
  """

  def __init__(self, xv, fxv, dfxv=None, xg=None, fpxg=None, dfpxg=None, \
               N=None, l=1, verbose=1, safety_factor=1.0):
    """
    __init__(self, xv, fxv, dfxv=None, xg=None,
             fpxg=None, dfpxg=None, N=None, l=1)
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
    Note: beta and gamma cannot be user-specified.
      They are automatically computed at every point
      during initializtion (can take a while).
    """

    Interp2D.__init__(self, xv, fxv, dfxv, xg, fpxg, dfpxg, 1, 1, \
                      N, l, verbose, safety_factor)
    # calculate gammas and construct interpolation object for gamma
    self.beta = self.calc_beta()
    self.gamma = self.calc_gamma()
    self.log_gamma_interp = Interp2D(xv, log(self.gamma), verbose=0, \
                                     safety_factor=100)



  def calc_gamma(self):
    """
    Estimate the `wave number' parameter gamma at
      each point from the data points.
    """
    assert isinstance(self.beta, float)
    if self.verbose > 1:
      print '    calculating gamma...'

    gamma_min_all, gamma_max_all = self._calc_gamma_bounds()
    # logorithmic bisection for gamma
    gamma = zeros(self.nv)
    for iv in range(self.nv):
      gamma_min, gamma_max = gamma_min_all, gamma_max_all
      while gamma_max / gamma_min > 1.1:
        if self.verbose > 2:
          print '    bisecting [', gamma_min, ',', gamma_max, '] for gamma...'
        gamma_mid = sqrt(gamma_max * gamma_min)
        res_ratio = self._calc_res_ratio(iv, self.beta, gamma_mid)
        if res_ratio < 1.0:
          gamma_max = gamma_mid
        else:
          gamma_min = gamma_mid
      # final selected gamma
      gamma_mid = sqrt(gamma_max * gamma_min)
      if self.verbose > 1:
        print '    using gamma = %f at point %d' % (gamma_mid, iv)
      gamma[iv] = gamma_mid
    return gamma



  def interp_matrices(self, x, beta=None, gamma=None):
    """
    See Interp2D.interp_matrices.
    This is the variable gamma version.
    """
    if beta is None:
      beta = self.beta
    if gamma is None:
      gamma = exp(self.log_gamma_interp.interp(x))
    return Interp2D.interp_matrices(self, x, beta, gamma)



  def grad_coef(self, x, beta=None, gamma=None):
    """
    See Interp2D.grad_coef.
    This is the variable gamma version.
    """
    if beta is None:
      beta = self.beta
    if gamma is None:
      gamma = exp(self.log_gamma_interp.interp(x))
    return Interp2D.grad_coef(self, x, beta, gamma)
