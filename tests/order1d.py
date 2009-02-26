"""
Qiqi Wang  January 2009  Sample use of interp1d.py
"""

import sys

import numpy
import pylab
from numpy import zeros, ones, linspace, kron, linalg, exp, sqrt, diag, \
                  arctan, pi, abs

sys.path.extend(['.', '..', 'tests'])
from wanginterp import Interp1D, Interp1DVG

pylab.matplotlib.interactive(True)

# ========================= Tune this =========================

#CASE = 'cos'
#def func(x):
#  f = numpy.cos(x)
#  return f
#
CASE = 'runge'
def func(x):
  f = 1.0 / (1.0 + x**2)
  return f

# CASE = 'notch'
# def func(x):
#   f = numpy.cos(x)
#   f -= 2*exp(-(x*4)**2)
#   return f
# 
# CASE = 'step'
# def func(x):
#   f = numpy.cos(x)
#   f[x<0] *= -1
#   return f

ns = range(4, 64, 8)  # number of data points
l2, li, l2vg, livg = [], [], [], []

for n in ns:
  print 'n = ', n
  y = linspace(-5, 5, 250)
  fye = func(y)
  x = linspace(-5, 5, n)
  fx = func(x)
  
  # --------- use of Interp1D class ----------
  interp = Interp1D(x, fx, p=1, verbose=1)
  fy, sig = interp.interp(y, compute_df=True)
  # ------------------------------------------
  
  l2.append(sqrt(((fy-fye)**2).sum() / y.size))
  li.append(abs(fy - fye).max())
  
  # --------- use of Interp1DVG class ----------
  interp = Interp1DVG(x, fx, p=1, verbose=1)
  fy, sig = interp.interp(y, compute_df=True)
  # ------------------------------------------
  
  l2vg.append(sqrt(((fy-fye)**2).sum() / y.size))
  livg.append(abs(fy - fye).max())

pylab.figure(figsize=(10,6))
pylab.semilogy(ns, l2, 'b-o')
pylab.semilogy(ns, li, 'b:o', mfc='w')
pylab.semilogy(ns, l2vg, 'r-o')
pylab.semilogy(ns, livg, 'r:o', mfc='w')

raw_input('Push enter to continue...')

