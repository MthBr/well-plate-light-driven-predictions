

from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32


image_shape=[128.0, 256.0]

out  = round(log(min(image_shape)) / log(2) - 1)



from sympy import *
sympy.latex(eval(round(log(min(image_shape)) / log(2) - 1))) 