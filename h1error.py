#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:56:55 2019

@author: maged
"""

import numpy as np
import scipy.linalg as la

from funs import *

def H1_error(element_linear_num,u):
        
        node_linear_num = element_linear_num + 1
        
        
        a = 0
        b = 1.0

        grid = np.linspace ( a, b, node_linear_num )

        h1_error = 0.0

        # u_mat is the solution u in each matrix 
        u_mat = np.zeros((node_linear_num,node_linear_num))

        v = 0
        for j in range ( 0, node_linear_num ):
            for i in range ( 0, node_linear_num ):
                u_mat[i,j] = u[v]
                v = v + 1

        # Quadrature defination
        quad_num = 3
        
        quad_point = np.array (( \
                                -0.774596669241483377035853079956, \
                                 0.0, \
                                 0.774596669241483377035853079956 ) )
        
        quad_weight = np.array (( \
                                 5.0 / 9.0, \
                                 8.0 / 9.0, \
                                 5.0 / 9.0 ))
        
    
        
        for ex in range ( 0, element_linear_num ):
        
           w = ex
           e = ex + 1
        
           xw = grid[w]
           xe = grid[e]
        
           for ey in range ( 0, element_linear_num ):
        
             s = ey
             n = ey + 1
        
             ys = grid[s]
             yn = grid[n]
        
             sw =   ey       * node_linear_num + ex
             se =   ey       * node_linear_num + ex + 1
        
             nw = ( ey + 1 ) * node_linear_num + ex
             ne = ( ey + 1 ) * node_linear_num + ex + 1
        
        #
        #  The 2D quadrature rule is the 'product' of X and Y copies of the 1D rule.
        #
             for qx in range (0, quad_num):
                 xq = (( 1.0 - quad_point[qx] ) * xw
                     + ( 1.0 + quad_point[qx] ) * xe ) / 2.0
        
                 for qy in range(0,quad_num):
                     yq = (( 1.0 - quad_point[qy] ) * ys
                         + ( 1.0 + quad_point[qy] ) * yn ) / 2.0
        
                     wq = quad_weight[qx] * quad_weight[qy] * (xe - xw) / 2.0 * (yn - ys) / 2.0
        
        #
        #  Evaluate all four basis functions, and their X and Y derivatives.
        #
                     vswx = (    -1.0 ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
                     vswy = ( xe - xq ) / ( xe - xw ) * (    -1.0 ) / ( yn - ys )

                     vsex = ( 1.0     ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
                     vsey = ( xq - xw ) / ( xe - xw ) * (    -1.0 ) / ( yn - ys )
                          
                     vnwx = (    -1.0 ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys ) 
                     vnwy = ( xe - xq ) / ( xe - xw ) * ( 1.0     ) / ( yn - ys ) 
                 
                     vnex = ( 1.0     ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys )
                     vney = ( xq - xw ) / ( xe - xw ) * ( 1.0     ) / ( yn - ys )
                 
                     uxq = u_mat[w,s] * vswx + u_mat[e,s] * vsex + u_mat[w,n] * vnwx + u_mat[e,n] * vnex
                     uyq = u_mat[w,s] * vswy + u_mat[e,s] * vsey + u_mat[w,n] * vnwy + u_mat[e,n] * vney             
                
            
                     exq = exact_fnx(xq, yq)
                     eyq = exact_fny(xq, yq)
             
                     h1_error = h1_error + wq * ( ( uxq - exq ) * ( uxq - exq ) +  ( uyq - eyq ) * (uyq - eyq) )   
    
    
        h1_error = np.sqrt(h1_error)
        return h1_error