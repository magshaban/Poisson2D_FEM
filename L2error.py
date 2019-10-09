#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:20:26 2019

@author: maged
"""
import numpy as np
import scipy.linalg as la

from funs import *

def L2_error(element_linear_num,u):
        
        node_linear_num = element_linear_num + 1
        
        
        a = 0
        b = 1.0

        grid = np.linspace ( a, b, node_linear_num )

        e2 = 0.0

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
        
        #
        #  x and y for each node.
        #
#        x = np.zeros( node_linear_num * node_linear_num)
#        y = np.zeros( node_linear_num * node_linear_num)
#        
#        v = 0
#        for j in range (0, node_linear_num):
#           for i in range (0, node_linear_num):
#               x[v]= grid[i]
#               y[v] = grid[j]
#               v = v + 1
        
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
                     vsw  = ( xe - xq ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
                     vse  = ( xq - xw ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
                     vnw  = ( xe - xq ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys )
                     vne  = ( xq - xw ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys )
        
                     uq = u_mat[w,s] * vsw + u_mat[e,s] * vse + u_mat[w,n] * vnw + u_mat[e,n] * vne
                     #print(w,n,e,s)
                     eq = exact_fn(xq, yq)
        
                     e2 = e2 + wq * (uq - eq) *  (uq - eq)    
    
    
        e2 = np.sqrt(e2)
        return e2
 #print('\n   The L2 error = ', e2)   
    
    