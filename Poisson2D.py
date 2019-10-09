#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Solves Poission 2D equation using Finite element method
# This is the Main file <Poisson2D.py>. 
#
#  -To calculate the Sloution L2 error for diffrent spacing h use <l2error_plot.py>.
#  -The file <solver.py> contains a simple solver for this problem. 
#  -The file <l2error.py> is the solution l2 error evaluation function. 
#
#  Problem description:
#
#    The PDE is defined for 0 < x < 1, 0 < y < 1:
#      - uxx - uyy = f(x)
#    with boundary conditions
#      u(0,y) = 0,
#      u(1,y) = 0,
#      u(x,0) = 0,
#      u(x,1) = 0.
#
#    The exact solution is:
#      exact(x) = x * ( 1 - x ) * y * ( 1 - y ).
#    The right hand side f(x) is:
#      f(x) = 2 * x * ( 1 - x ) + 2 * y * ( 1 - y )
#
#  Modified: Q1 instead using P1 in the pervious code. 
#    The unit square is divided into N by N squares.  Bilinear finite 
#    element basis functions are defined, and the solution is sought as a
#    piecewise linear combination of these basis functions.
#
# Author: Maged Shaaban 
# magshaban@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from quad import *
from funs import *
from l2error import *

element_linear_num = 20
node_linear_num = element_linear_num + 1
element_num = element_linear_num * element_linear_num
node_num = node_linear_num * node_linear_num

a = 0
b = 1.0

grid = np.linspace ( a, b, node_linear_num )

#print( '' )
#print( '  Nodes along x axis:' )
#print( '' )
#
##
## to be modified from 1 to node_linear_num+1 wich gives 1,2,3,4,...
##
#for i in range ( 0, node_linear_num ):
#   print( '  %d  %f' %( i, grid[i] ) )

#
#  Set up a quadrature rule.
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
 

#
#to plot the mass matrix before adding the boundary conditions 
             
#             
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('The Stiffness Matrix')
ax1.matshow(A_in)
ax2.spy(A_in)
plt.show()

#     
#to plot the mass matrix 
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('The Stiffness Matrix with BC contribution')
ax1.matshow(A)
ax2.spy(A)
plt.show()




fig = plt.figure()
ax = fig.gca(projection='3d')
  
# Make data.
z =u
# Plot the surface.
surf = ax.plot_trisurf(x, y, z, cmap=cm.Spectral,linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.7, aspect=9)
plt.title('The solution $U$')  
plt.show()

from matplotlib.colors import LogNorm

x_list = x
y_list = y
z_list = u
N = int(len(u) ** .5)
z = u.reshape(N, N)
plt.title('The solution $U$ from above')
plt.imshow(z, extent=(np.amin(x_list),
                      np.amax(x_list),
                      np.amin(y_list),
                      np.amax(y_list)),
        norm=LogNorm(), aspect='auto')
#plt.colorbar()
plt.show()


print('#####################################################')
print( '''  The Exact Solution 
      U_exact =  xy(1-x)(1-y)
  evaluated on each node is ''')
print('#####################################################')

#############################
#
# The Exact solution 
#

fig = plt.figure()
ax = fig.gca(projection='3d')
  
## Make data.
z = u_exact
#############################
#
# The Exact solution 
#
# Plot the surface.
surf = ax.plot_trisurf(x, y, z, cmap=cm.Spectral,linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.7, aspect=9)
plt.title('The exact solution $U_{exact}$')
plt.show()

#
# Print the elements, listing the nodes in counterclockwise order.
#
#if(True):
#   e = 0
#   print ( '' )
#   print ( '   The elements, listing the nodes in counterclockwise order' )
#   print ( '' )
#   for j in range ( 0, element_linear_num ):
#      y = grid[j]
#      for i in range ( 0, element_linear_num ):
#          sw =   j       * node_linear_num + i
#          se =   j       * node_linear_num + i + 1
#          nw = ( j + 1 ) * node_linear_num + i
#          ne = ( j + 1 ) * node_linear_num + i + 1
#          print ( '%4d  %4d  %4d  %4d' % ( sw, se, ne, nw ) )
#          e = e + 1
#
xerror = np.zeros(node_linear_num*node_linear_num)
error = np.zeros(node_linear_num*node_linear_num)
v = 0
for j in range ( 0, node_linear_num ):
     for i in range ( 0, node_linear_num ):
             error [v]= u[v]- u_exact[v]
             xerror[v]= v
             v = v + 1

 
plt.plot(xerror,error,'b')
plt.xlabel('Node')
plt.ylabel('$Error$')
plt.grid()
plt.show()   

fig = plt.figure()
ax = fig.gca(projection='3d')
z =error

surf = ax.plot_trisurf(x, y, z, cmap=cm.Spectral,linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('  %.05f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.7, aspect=9)
plt.title('$Error$')
plt.show()

# To evaluate the L2 error 
L2_error = L2_error(element_linear_num, u)
print('\n\n\n#####################################################')
print('\n        The L2 error = ', L2_error)
print('\n#####################################################')

#
#  Terminate.
#
print ( '' )
print ( '  >> Normal end of execution.' )
