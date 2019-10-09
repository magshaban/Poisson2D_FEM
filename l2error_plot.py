#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:42:05 2019

@author: maged

part of <poisson2D.py> 


"""
import numpy as np 

from solver import * 
from l2error import *


start = 15
stop = 60
step = 2

v = 0
for i in range(start,stop,step):
    v = v + 1
    
e2 = np.zeros(v)
h = np.zeros(v)
node_num = np.zeros(v)
elements = np.zeros(v)

print ( '' )
print ( '   Total Nodes    Total Elements           h               L2_error' )
print ( '' )

v = 0
for i in range(start,stop,step):
    elements[v]= i * i
    u,h[v],node_num[v] = solver(element_linear_num = i)
    e2[v] = L2_error(element_linear_num = i,u = u)
    print ( '      %4d            %4d              %8f        %14g' 
           % ( node_num[v],elements[v], h[v], e2[v] ) )
    v = v + 1 

print ( '' )

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