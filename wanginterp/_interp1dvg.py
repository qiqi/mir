import copy
import sys
import pickle
import time

import numpy
import pylab
from numpy import zeros, ones, eye, kron, linalg, dot, exp, sqrt, diag, pi, \
                  asarray, sign

from _interp1d import Interp1D




# ---------------------------------------------------------------------------- #




class Interp1DVG(Interp1D):
  """
  Variable gamma version of Interp1D.
  """

  def __init__(self, xv, fxv, dfxv=None, xg=None, fpxg=None, dfpxg=None, \
               N=None, l=1, verbose=1):
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

    Interp1D.__init__(self, xv, fxv, dfxv, xg, fpxg, dfpxg, 1, 1, \
                      N, l, verbose)
    # calculate gammas and construct interpolation object for gamma
    self.beta = self.calc_beta()
    self.gamma = self.calc_gamma()
    self.gamma_interp = Interp1D(xv, self.gamma, verbose=0)



  def _calc_res_ratio(self, iv, beta, gamma):
    """
    A utility function used by calc_gamma, calculates
      the ratio of real residual to the estimated
      residual at the iv'th value data point, which
      is used to make decision in the bisection process
      for gamma at the iv'th data point.
    """
    base = range(iv) + range(iv+1,self.nv)
    subinterp = Interp1D(self.xv[base], self.fxv[base], self.dfxv[base], \
                         self.xg, self.fpxg, self.dfpxg, beta, gamma, \
                         self.N, self.l, self.verbose)
    av, ag, er2 = subinterp.interp_coef(self.xv[iv])
    resid = dot(av,self.fxv[base]) + dot(ag,self.fpxg) - self.fxv[iv]
    return resid**2 / (er2 + self.dfxv[iv]**2) * 1.0E+1



  def calc_gamma(self):
    """
    Estimate the `wave number' parameter gamma at
      each point from the data points.
    """
    assert isinstance(self.beta, float)
    # upper and lower bounds
    sorted_xv = numpy.array(sorted(self.xv))
    sorted_xg = numpy.array(sorted(self.xg))
    if self.xg.size == 0:
      delta_max = self.xv.max() - self.xv.min()
      delta_min = (sorted_xv[1:] - sorted_xv[:-1]).min()
    else:
      delta_max = max(self.xv.max(), self.xg.max()) - \
                  min(self.xv.min(), self.xg.min())
      delta_min = min((sorted_xv[1:] - sorted_xv[:-1]).min(), \
                      (sorted_xg[1:] - sorted_xg[:-1]).min())
    assert delta_max > delta_min
    gamma_min_all = 1. / delta_max
    gamma_max_all = pi / delta_min
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
    See Interp1D.interp_matrices.
    This is the variable gamma version.
    """
    if beta is None:
      beta = self.beta
    if gamma is None:
      gamma = self.gamma_interp.interp(x)
    return Interp1D.interp_matrices(self, x, beta, gamma)



  def grad_coef(self, x, beta=None, gamma=None):
    """
    See Interp1D.grad_coef.
    This is the variable gamma version.
    """
    if beta is None:
      beta = self.beta
    if gamma is None:
      gamma = self.gamma_interp.interp(x)
    return Interp1D.grad_coef(self, x, beta, gamma)
