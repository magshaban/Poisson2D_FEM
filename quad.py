#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maged
"""


#
#  Set up a quadrature rule.
#  This rule is defined on the reference interval [0,1].
#
import numpy as np 

def quad():
    quad_num = 3

    quad_point = np.array (( \
                            0.112701665379258311482073460022, \
                            0.5, \
                            0.887298334620741688517926539978 ) )

    quad_weight = np.array (( \
                             5.0 / 18.0, \
                             8.0 / 18.0, \
                             5.0 / 18.0 ))
    return quad_num, quad_point,quad_weight
#