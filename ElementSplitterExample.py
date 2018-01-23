#!/bin/env python
# -*- coding: utf-8 -*-
import mesh
import splitter
import numpy as np
import VisuSplitMesh as VSM
from mpi4py import MPI

meshName = "Carre.msh"
#meshName = "L.msh"

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

if 1 == size :
    print "Lancer le script en parallele avec plus d'un processus !"
    exit(-1)
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

etsDomains = splitter.splitEltMesh( nbDoms, nbVerts, elt2verts )
nbElts_loc = etsDomains[rank].shape[0]
fich.write("Nombre d'éléments contenus dans le domaine : %d\n"%nbElts_loc)

if rank == 0 :
    elt2doms = np.zeros((nbElts,), np.double)
    ia = 0.
    for a in etsDomains :
        for e in a :
            elt2doms[e] = ia
        ia += 1

    # Calcul l'interface :
    ie = 0
    mask = np.array([-1,]*nbVerts, np.short)
    for e in elt2verts :
        d = elt2doms[ie]
        if mask[e[0]] == -1 :
            mask[e[0]] = d
        elif mask[e[0]] != d :
            mask[e[0]] = -2
        if mask[e[1]] == -1 :
            mask[e[1]] = d
        elif mask[e[1]] != d :
            mask[e[1]] = -2
        if mask[e[2]] == -1 :
            mask[e[2]] = d
        elif mask[e[2]] != d :
            mask[e[2]] = -2
        ie += 1

    nbInterf = 0
    for m in mask :
        if m == -2 :
            nbInterf += 1

    interfNodes = np.empty(nbInterf, np.long)
    nbInterf = 0
    for im in xrange(mask.shape[0]):
        if mask[im] == -2 :
            interfNodes[nbInterf] = im
            nbInterf += 1

    VSM.view( coords, elt2verts, nbDoms, elt2doms, indInterfNodes = interfNodes, title='Partition par elements',visuIndElts = False, visuIndVerts = False)

# Fabrication des coordonnées locales et de elt2verts local :

eltDomain = etsDomains[rank]
# Construction d'un loc2glob à partir des éléments locaux :
#     0. On construit un masque :
mskNodes = np.zeros(nbVerts, np.short)
for ie in eltDomain:
    nds = elt2verts[ie,:]
    mskNodes[nds[0]] = 1
    mskNodes[nds[1]] = 1
    mskNodes[nds[2]] = 1
#     1. On compte le nombre de noeuds contenus dans le maillage :
nbVerts_loc = mskNodes.nonzero()[0].shape[0]
loc2glob = np.array([ iv for iv in xrange(nbVerts) if 1==mskNodes[iv]],np.long)
fich.write("Nombre de noeuds locaux : %d\n"%nbVerts_loc)

# construction de glob2loc :
glob2loc = -np.ones(nbVerts, np.long)
iv = 0
for v in loc2glob :
    glob2loc[v] = iv
    iv += 1

fich.write("Numérotation loc2glob : %s\n"%repr(loc2glob))
fich.write("Numérotation glob2loc : %s\n"%repr(glob2loc))

# Extraction des connections locales elt2verts_loc :
elt2verts_loc = np.empty((nbElts_loc,3), np.long)
iloc = 0
for im in eltDomain:
    elt2verts_loc[iloc,0] = glob2loc[elt2verts[im,0]]
    elt2verts_loc[iloc,1] = glob2loc[elt2verts[im,1]]
    elt2verts_loc[iloc,2] = glob2loc[elt2verts[im,2]]
    iloc += 1

fich.write("Elt2Verts local : %s\n"%repr(elt2verts_loc))

# Extraction des coordonnées locales :
coords_loc = np.empty( (nbVerts_loc, 4), np.double)
for ip in xrange(loc2glob.shape[0]):
    coords_loc[ip,:] = coords[loc2glob[ip],:]

partition = rank*np.ones( nbVerts_loc, np.long )

VSM.view(coords_loc,elt2verts_loc,nbDoms,partition,
         title='Partition locale proc %d'%rank)
