#!/bin/env python
# -*- coding: utf-8 -*-
import mesh
import fem
import splitter
import numpy as np
import VisuSplitMesh as VSM
from mpi4py import MPI

meshName = "Carre.msh"
#meshName = "L.msh"

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

nbDoms = size
fich = open("output%03d"%rank, 'w')

fich.write("Décomposition sur %d domaines : je suis le domaine %d\n"%(nbDoms, rank))

m = mesh.read(meshName)
coords    = m[0]
elt2verts = m[1]
nbVerts = coords.shape[0]
nbElts  = elt2verts.shape[0]

fich.write("elt2verts global : %s\n"%repr(elt2verts))
begVert2Elts, vert2elts = mesh.compvert2elts(elt2verts)

begCols, indRows = fem.comp_skel_csc_mat(elt2verts, (begVert2Elts, vert2elts) )

ndsDomains = splitter.splitNodeMesh( nbDoms, (begCols, indRows) )
ndsDomain = ndsDomains[rank]

fich.write("Nombre de noeuds locaux : %04d\n"%ndsDomain.shape[0])

if rank == 0 :
    vert2dom = np.zeros((nbVerts,), np.double)
    ia = 0.
    for a in ndsDomains :
        for v in a :
            vert2dom[v] = ia
        ia += 1

    
    mask = np.zeros((nbVerts,), np.short)
    for e in elt2verts :
        d1 = vert2dom[e[0]]
        d2 = vert2dom[e[1]]
        d3 = vert2dom[e[2]]
        if (d1 != d2) or (d1 != d3) or (d2 != d3) :
            mask[e[0]] = 1
            mask[e[1]] = 1
            mask[e[2]] = 1

    nbInterf = 0
    for m in mask :
        if m == 1 :
            nbInterf += 1
    interfNodes = np.empty(nbInterf, np.long)
    nbInterf = 0
    for im in xrange(mask.shape[0]):
        if mask[im] == 1 :
            interfNodes[nbInterf] = im
            nbInterf += 1

    VSM.view( coords, elt2verts, nbDoms, vert2dom, indInterfNodes = interfNodes, title='Partition par sommets',visuIndElts = False, visuIndVerts = False)

nbVerts_loc = ndsDomain.shape[0]
loc2glob = ndsDomain.shape[0]

# construction de glob2loc :
glob2loc = -np.ones(nbVerts, np.long)
iv = 0
for v in loc2glob :
    glob2loc[v] = iv
    iv += 1

fich.write("Numérotation loc2glob : %s\n"%repr(loc2glob))
fich.write("Numérotation glob2loc : %s\n"%repr(glob2loc))

mask = np.zeros( nbElts, np.short)
for ptElt in xrange(nbVerts) :
    for e in vert2elts[begVert2Elts[ptElt]:begVert2Elts[ptElt+1]] :
        mask[e] = 1

nbEltsLoc = mask.nonzero()[0].shape[0]
elt2verts_loc = np.array(nbEltsLoc, 3)
indElt = 0
for i in xrange(mask.shape[0]):
    if mask[i] == 1 :
        elt2verts_loc[indElt,0] = glob2loc[elt2verts[i,0]]
        elt2verts_loc[indElt,1] = glob2loc[elt2verts[i,1]]
        elt2verts_loc[indElt,2] = glob2loc[elt2verts[i,2]]

