#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:38:59 2019

@author: maged
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from quad import *
from funs import *

element_linear_num = 32
node_linear_num = element_linear_num + 1
element_num = element_linear_num * element_linear_num
node_num = node_linear_num * node_linear_num

a = 0
b = 1.0

grid = np.linspace ( a, b, node_linear_num )

h1_error = 0.0

#
quad_num, quad_point, quad_weight = quad()
#
#  x and y for each node.
#
x = np.zeros( node_linear_num * node_linear_num)  
y = np.zeros( node_linear_num * node_linear_num)  

v = 0
for j in range (0, node_linear_num):   
   for i in range (0, node_linear_num):              
       x[v]= grid[i]
       y[v] = grid[j]
       v = v + 1


#

#
# Memory allocation.
#
A = np.zeros((node_num, node_num))
rhs = np.zeros(node_num)

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
         xq = xw + quad_point[qx] * (xe - xw)
         
         for qy in range(0,quad_num):
             yq = ys + quad_point[qy] * (yn - ys)
             wq = quad_weight[qx] * quad_weight[qy] * (xe - xw) * (yn - ys)     
#
#  Evaluate all four basis functions, and their X and Y derivatives.
#
             vsw  = ( xe - xq ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
             vswx = (    -1.0 ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
             vswy = ( xe - xq ) / ( xe - xw ) * (    -1.0 ) / ( yn - ys )

             vse  = ( xq - xw ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
             vsex = ( 1.0     ) / ( xe - xw ) * ( yn - yq ) / ( yn - ys )
             vsey = ( xq - xw ) / ( xe - xw ) * (    -1.0 ) / ( yn - ys )
              
             vnw  = ( xe - xq ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys ) 
             vnwx = (    -1.0 ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys ) 
             vnwy = ( xe - xq ) / ( xe - xw ) * ( 1.0     ) / ( yn - ys ) 
              
             vne  = ( xq - xw ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys )
             vnex = ( 1.0     ) / ( xe - xw ) * ( yq - ys ) / ( yn - ys )
             vney = ( xq - xw ) / ( xe - xw ) * ( 1.0     ) / ( yn - ys )
#   
#  Compute contributions to the stiffness matrix.
#
             A[sw,sw] = A[sw,sw] + wq * ( vswx * vswx + vswy * vswy )
             A[sw,se] = A[sw,se] + wq * ( vswx * vsex + vswy * vsey )
             A[sw,nw] = A[sw,nw] + wq * ( vswx * vnwx + vswy * vnwy )
             A[sw,ne] = A[sw,ne] + wq * ( vswx * vnex + vswy * vney )
             rhs[sw]   = rhs[sw] + wq *   vsw  * rhs_fn ( xq, yq )
             
             A[se,sw] = A[se,sw] + wq * ( vsex * vswx + vsey * vswy )
             A[se,se] = A[se,se] + wq * ( vsex * vsex + vsey * vsey )
             A[se,nw] = A[se,nw] + wq * ( vsex * vnwx + vsey * vnwy )
             A[se,ne] = A[se,ne] + wq * ( vsex * vnex + vsey * vney )
             rhs[se]   = rhs[se] + wq *   vse  * rhs_fn ( xq, yq )
              
             A[nw,sw] = A[nw,sw] + wq * ( vnwx * vswx + vnwy * vswy )
             A[nw,se] = A[nw,se] + wq * ( vnwx * vsex + vnwy * vsey )
             A[nw,nw] = A[nw,nw] + wq * ( vnwx * vnwx + vnwy * vnwy )
             A[nw,ne] = A[nw,ne] + wq * ( vnwx * vnex + vnwy * vney )
             rhs[nw]   = rhs[nw] + wq *   vnw  * rhs_fn ( xq, yq )
              
             A[ne,sw] = A[ne,sw] + wq * ( vnex * vswx + vney * vswy )
             A[ne,se] = A[ne,se] + wq * ( vnex * vsex + vney * vsey )
             A[ne,nw] = A[ne,nw] + wq * ( vnex * vnwx + vney * vnwy )
             A[ne,ne] = A[ne,ne] + wq * ( vnex * vnex + vney * vney )
             rhs[ne]   = rhs[ne] + wq *   vne  * rhs_fn( xq, yq )     
     
A_in = A       

#  Modify the linear system to enforce the boundary conditions where
#  X = 0 or 1 or Y = 0 or 1.
#
v = 0
for j in range ( 0, node_linear_num ):
    for i in range ( 0, node_linear_num ):

      if ( i == 0 or i == node_linear_num - 1 or j == 0 or j == node_linear_num - 1 ):
        A[v,0:node_num] = 0.0
        A[v,v] = 1.0
        rhs[v] = 0.0

      v = v + 1

#
#  Solve the linear system.
#
u = la.solve(A, rhs)

# u_mat is the solution u in each matrix 
u_mat = np.zeros((node_linear_num,node_linear_num))

v = 0
for j in range ( 0, node_linear_num ):
     for i in range ( 0, node_linear_num ):
         u_mat[i,j] = u[v]
         v = v + 1

print(u_mat)
u_exact = exact_fn(x, y)
   
#
#  Compare the solution and the error at the nodes.
#
print ( '' )
print ( '   Node    x         y              u               u_exact' )
print ( '' )
v = 0
for j in range ( 0, node_linear_num ):
     for i in range ( 0, node_linear_num ):
     
      print ( ' %4d  %8f  %8f  %14g  %14g' % ( v, x[v], y[v], u[v], u_exact[v] ) )
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
x = np.zeros( node_linear_num * node_linear_num)  
y = np.zeros( node_linear_num * node_linear_num)  

v = 0
for j in range (0, node_linear_num):   
   for i in range (0, node_linear_num):              
       x[v]= grid[i]
       y[v] = grid[j]
       v = v + 1
       
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
print('\n   The H1 error = ', h1_error)
