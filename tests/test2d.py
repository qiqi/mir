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
Qiqi Wang  January 2009  Sample use of interp1d.py
"""

import sys

import pylab
import numpy
from numpy import zeros, ones, linspace, kron, linalg, exp, sqrt, diag, \
                  arctan, pi, log10, asarray, floor

pylab.matplotlib.interactive(True)
sys.path.extend(['.', '..', 'tests'])

import misc
from wanginterp import Interp2D, Interp2DVG

# ========================== Tune this ==========================

# def func(x):
#   return x[:,0]**2 + x[:,1]**2
# 
# def func(x):
#   return x[:,0] + x[:,1]
# 
def func(x):
  return 1.0 / (x[:,0]**2 + x[:,1]**2 + 1.0)

n = 56  # number of data points
x = (misc.niederreiter_2d[:n,:] * 2 - 1) * 2

# ========================== Tune this ==========================

ny = 20
y1d = linspace(-2, 2, ny)
y = misc.cartesian_2d(y1d)

fx, fy = func(x), func(y)

# ---------- use of Interp2D class --------
print
print 'Using Interp2D class:'
print
interp = Interp2D(x, fx, p=1, verbose=1)
f, sig = interp.interp(y, compute_df=True)
gamma = interp.gamma * ones([len(y1d), len(y1d)])

pylab.figure(figsize=(8,6))
pylab.subplot(2,2,1)
misc.contourplot(y1d, f, x, 'interpolant')
pylab.subplot(2,2,2)
misc.contourplot(y1d, sig, x, 'estimated residual')
pylab.subplot(2,2,3)
misc.contourplot(y1d, abs(f-fy), x, 'true residual')
pylab.subplot(2,2,4)
misc.contourplot(y1d, gamma, x, 'gamma')

linf = abs(f-fy).max()
l2 = sqrt(((f-fy)**2).sum() / f.size)

print "Number of points: %d; L-infinity error: %f; L-2 error: %f" % \
      (n, linf, l2)
print
raw_input('Push enter to continue...')

# ---------- use of Interp2DVG class --------
print
print 'Using Interp2DVG class:'
print
interp = Interp2DVG(x, fx, p=1, verbose=1)
f, sig = interp.interp(y, compute_df=True)
gamma = exp(interp.log_gamma_interp.interp(y))

pylab.figure(figsize=(8,6))
pylab.subplot(2,2,1)
misc.contourplot(y1d, f, x, 'interpolant')
pylab.subplot(2,2,2)
misc.contourplot(y1d, sig, x, 'estimated residual')
pylab.subplot(2,2,3)
misc.contourplot(y1d, abs(f-fy), x, 'true residual')
pylab.subplot(2,2,4)
misc.contourplot(y1d, gamma, x, 'gamma')

linf = abs(f-fy).max()
l2 = sqrt(((f-fy)**2).sum() / f.size)

print "Number of points: %d; L-infinity error: %f; L-2 error: %f" % \
      (n, linf, l2)
print
raw_input('Push enter to continue...')

