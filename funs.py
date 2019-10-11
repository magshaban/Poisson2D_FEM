#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 23:47:19 2019

@author: maged
"""
# This file contains the RHS functions and the exact solutions

#
# The RHS function f(x)
#
def rhs_fn(x, y):

  value = 2.0 * x * ( 1.0 - x ) + 2.0 * y * ( 1.0 - y )
  return value


#
# The exact solution 
#  
def exact_fn(x, y):
  value = x * ( 1.0 - x ) * y * ( 1.0 - y )
  return value


#
# The exact solution 
#  
def exact_fnx(x, y):
  value = ( 1.0 - 2.0 * x ) * y * ( 1.0 - y )
  return value    


#
# The exact solution 
#  
def exact_fny(x, y):
  value = x * ( 1.0 - x ) *  ( 1.0 - 2.0 * y )
  return value