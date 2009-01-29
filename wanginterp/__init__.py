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
wanginterp verson 1.0.0
Qiqi Wang  January 2009


Main classes:

Interp1D: Interpolation and regression in one-dimensional space
Interp2D: Interpolation and regression in two-dimensional space
InterpND: Interpolation and regression in multi-dimensional space
Interp1DVG: Interpolation and regression in one-dimensional space
            with variable gamma.

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
from _interp2dvg import Interp2DVG
from _interpnd import InterpND
