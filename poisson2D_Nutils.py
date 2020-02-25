#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:11:32 2020

@author: maged
"""


import nutils, numpy
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


nelems = 10
topo, geom = nutils.mesh.rectilinear([numpy.linspace(0, 1, nelems+1), numpy.linspace(0, 1, nelems+1)])

ns = nutils.function.Namespace()
ns.x = geom
ns.basis = topo.basis('std', degree=1)
ns.u = 'basis_n ?lhs_n'

res  = topo.integral('basis_n,i u_,i d:x' @ ns, degree=2)
# res -= topo.integral('(basis_n (2 x_0 (1 - x_0) + 2 x_1 (1 - x_1))) d:x' @ ns, degree=2)
inertia = topo.integral('basis_n u d:x' @ ns, degree=5)


sqrinit = topo.integral('(u - exp(-?y_i ?y_i)(y_i = 5 (x_i - 0.5_i)))^2 d:x' @ ns, degree=5)
#sqrinit = topo.integral('(u)^2 d:x' @ ns, degree=5)

lhs0 = nutils.solver.optimize('lhs', sqrinit)

sqr  = topo.boundary['left'].integral('u^2 d:x' @ ns, degree=2)
sqr += topo.boundary['right'].integral('u^2 d:x' @ ns, degree=2)
sqr += topo.boundary['top'].integral('u^2 d:x' @ ns, degree=2)
sqr += topo.boundary['bottom'].integral('u^2 d:x' @ ns, degree=2)


cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)


timestep = 1.
bezier = topo.sample('bezier', 7)
with treelog.iter.plain('timestep', nutils.solver.impliciteuler('lhs', res, inertia, timestep=timestep, lhs0=lhs0, constrain=cons, newtontol=1e-5)) as steps:
  for itime, lhs in enumerate(steps):
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
    export.triplot('solution.png', x, u, tri=bezier.tri, hull=bezier.hull, clim=(0,1))


#lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

#bezier = topo.sample('bezier', 8)
#x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)

#plt.tripcolor(x[:,0], x[:,1], bezier.tri, u, shading='gouraud', rasterized=True)
#plt.colorbar()
#plt.xlabel('x_0')
#plt.ylabel('x_1')


#def makevtk(ns,domain,dofs,filename):
#	sample = domain.sample('bezier',2)
#	x, u = sample.eval([ns.x, ns.u], lhs=dofs)
#	title = filename
#	nutils.export.vtk(title,sample.tri,x,solution=u)

#makevtk(ns,topo,dofs=lhs,filename='results')

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_trisurf(x[:,0], x[:,1], u, cmap=cm.Spectral,linewidth=0, antialiased=False)
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#fig.colorbar(surf, shrink=0.7, aspect=9)
#plt.title('Nutils solution')
#plt.xlabel('x_0')
#plt.ylabel('x_1') 
#print('hey')
#plt.show()
#print('bye')
