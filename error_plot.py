#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:42:05 2019

@author: maged

part of <poisson2D.py> 


"""
import numpy as np 
import matplotlib.pyplot as plt

from solver import * 
from l2error import *
from h1error import *


start = 2
stop = 7
step = 1

v = 0
for k in range(start,stop,step):
    v = v + 1
    
    
h1_error = np.zeros(v)    
e2 = np.zeros(v)
h = np.zeros(v)
node_num = np.zeros(v)
elements = np.zeros(v)

print ( '' )
print ( '   Total Nodes    Total Elements           h                L2_error                H1_error' )
print ( '' )

v = 0
for k in range(start,stop,step):
    i = 2**k
    print ('i=',i)
    elements[v]= i * i
    u,h[v],node_num[v] = solver(element_linear_num = i)
    e2[v] = L2_error(element_linear_num = i,u = u)
    h1_error[v] = H1_error(element_linear_num = i,u = u)
    print ( '      %4d            %4d              %8f        %14g        %14g' 
           % ( node_num[v],elements[v], h[v], e2[v], h1_error[v] ) )
    v = v + 1 

print ( '' )

#
# plotting L2 error 
#
plt.plot(node_num,e2,'b')
plt.xlabel('Nodes')
plt.ylabel('$L_2 \,\,\, Error$')
plt.grid()
plt.title('Solution error in the $L_2$ norm\n with the total number of nodes')
plt.show()  


plt.plot(elements,e2,'b')
plt.xlabel('Elements')
plt.ylabel('$L_2 \,\,\, Error$')
plt.grid()
plt.title('Solution error in the $L_2$ norm\n with the total number of elements')
plt.show() 

plt.plot(h,e2,'r')
plt.xlabel('h')
plt.ylabel('$L_2 \,\,\, Error$')
plt.grid()
plt.title('Solution error in the $L_2$ norm\n with the grid spacing $\Delta x =  \Delta y = h $')
plt.show()  

plt.plot(np.log2(h),np.log2(e2),'r')
plt.xlabel('log(h)')
plt.ylabel('$log(L_2 \,\,\, Error)$')
plt.grid()
plt.show()  

#
# plotting H1 error 
#
plt.plot(node_num,h1_error,'b')
plt.xlabel('Nodes')
plt.ylabel('$H_1 \,\,\, Error$')
plt.grid()
plt.title('Solution error in the $H_1$ semi-norm\n with the total number of nodes')
plt.show()  


plt.plot(elements,h1_error,'b')
plt.xlabel('Elements')
plt.ylabel('$H_1 \,\,\, Error$')
plt.grid()
plt.title('Solution error in the $H_1$ semi-norm\n with the total number of elements')
plt.show() 

plt.plot(h,h1_error,'r')
plt.xlabel('h')
plt.ylabel('$H_1 \,\,\, Error$')
plt.grid()
plt.title('Solution error in the $H_1$ semi-norm\n with the grid spacing $\Delta x =  \Delta y = h $')
plt.show()  

plt.plot(np.log2(h),np.log2(h1_error),'r')
plt.xlabel('log(h)')
plt.ylabel('$log(H_1 \,\,\, Error)$')
plt.grid()
plt.show() 

#
# To compare the errors  
#
plt.plot(np.log2(h),np.log2(e2),'b', label = '$log(L_2 \,\,\, Error)$')
plt.plot(np.log2(h),np.log2(h1_error),'r',label = '$log(H_1 \,\,\, Error)$')
plt.xlabel('log(h)')
plt.legend()
plt.grid()
plt.show() 


