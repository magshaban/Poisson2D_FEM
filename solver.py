#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: maged
"""

import numpy as np
import scipy.linalg as la

from quad import *
from funs import *


def solver(element_linear_num): 
                                                                                                                                
                                                                                                     
        node_linear_num = element_linear_num + 1                                                                                
        element_num = element_linear_num * element_linear_num                                                                   
        node_num = node_linear_num * node_linear_num                                                                            
                                                                                                                                
        a = 0                                                                                                                   
        b = 1.0                                                                                                                 
                                                                                                                                
        grid = np.linspace ( a, b, node_linear_num )   
        h = grid[1] - grid[0]                                                                 
                                                                                                                                
        e2 = 0.0                                                                                                                
                                                                                                                                
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
                x[v ]= grid[i]                                                                                                  
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
                                                                                                                                
                    for qy in range(0 ,quad_num):                                                                               
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
                        A[sw ,sw] = A[sw ,sw] + wq * ( vswx * vswx + vswy * vswy )                                              
                        A[sw ,se] = A[sw ,se] + wq * ( vswx * vsex + vswy * vsey )                                              
                        A[sw ,nw] = A[sw ,nw] + wq * ( vswx * vnwx + vswy * vnwy )                                              
                        A[sw ,ne] = A[sw ,ne] + wq * ( vswx * vnex + vswy * vney )                                              
                        rhs[sw]   = rhs[sw] + wq *   vsw  * rhs_fn ( xq, yq )                                                   
                                                                                                                                
                        A[se ,sw] = A[se ,sw] + wq * ( vsex * vswx + vsey * vswy )                                              
                        A[se ,se] = A[se ,se] + wq * ( vsex * vsex + vsey * vsey )                                              
                        A[se ,nw] = A[se ,nw] + wq * ( vsex * vnwx + vsey * vnwy )                                              
                        A[se ,ne] = A[se ,ne] + wq * ( vsex * vnex + vsey * vney )                                              
                        rhs[se]   = rhs[se] + wq *   vse  * rhs_fn ( xq, yq )                                                   
                                                                                                                                
                        A[nw ,sw] = A[nw ,sw] + wq * ( vnwx * vswx + vnwy * vswy )                                              
                        A[nw ,se] = A[nw ,se] + wq * ( vnwx * vsex + vnwy * vsey )                                              
                        A[nw ,nw] = A[nw ,nw] + wq * ( vnwx * vnwx + vnwy * vnwy )                                              
                        A[nw ,ne] = A[nw ,ne] + wq * ( vnwx * vnex + vnwy * vney )                                              
                        rhs[nw]   = rhs[nw] + wq *   vnw  * rhs_fn ( xq, yq )                                                   
                                                                                                                                
                        A[ne ,sw] = A[ne ,sw] + wq * ( vnex * vswx + vney * vswy )                                              
                        A[ne ,se] = A[ne ,se] + wq * ( vnex * vsex + vney * vsey )                                              
                        A[ne ,nw] = A[ne ,nw] + wq * ( vnex * vnwx + vney * vnwy )                                              
                        A[ne ,ne] = A[ne ,ne] + wq * ( vnex * vnex + vney * vney )                                              
                        rhs[ne]   = rhs[ne] + wq *   vne  * rhs_fn( xq, yq )                                                    
                                                                                                                                
                                                                                                                               
        #  Modify the linear system to enforce the boundary conditions where                                                    
        #  X = 0 or 1 or Y = 0 or 1.                                                                                            
        #                                                                                                                       
        v = 0                                                                                                                   
        for j in range ( 0, node_linear_num ):                                                                                  
            for i in range ( 0, node_linear_num ):                                                                              
                                                                                                                                
                if ( i == 0 or i == node_linear_num - 1 or j == 0 or j == node_linear_num - 1 ):                                
                    A[v ,0:node_num] = 0.0                                                                                      
                    A[v ,v] = 1.0                                                                                               
                    rhs[v] = 0.0                                                                                                
                                                                                                                                
                v = v + 1                                                                                                       
                                                                                                                                
        #                                                                                                                       
        #  Solve the linear system.                                                                                             
        #                                                                                                                       
        u = la.solve(A, rhs)                                                                                                    
                                                                                                                                 
        return u,h,node_num
    
# Test the function     
# u,h,node_num = solver(4)

#print('The solution u = ',u.T,'\n \n h =  ',  h,'\n \n The total number of nodes = ',node_num)