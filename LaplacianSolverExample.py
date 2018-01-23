#!/bin/env python
# -*- coding: utf-8 -*-
import mesh
import fem
import laplacian
import VisuSolution as VS
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from math import cos,sin,pi,sqrt

# fonction définissant les valeurs de la solution sur les conditions
# de Dirichlet :
def g(x,y) :
    return cos(2*pi*x)+sin(2*pi*y)

# Pour un autre maillage, changer le nom ici
#meshName = "Carre.msh"
meshName = "L.msh"

m = mesh.read(meshName)
# Récupération des coordonnées des sommets. Contient quatre dimensions,
# la quatrième permettant de "marquer" si un noeud est une condition limite
# ( ici uniquement de Dirichlet ) ou non, les trois autres étant x,y et z
# dans le sens usuel.
# coords est de dimension (nbre sommets, 4)
coords = m[0]
nb_verts = coords.shape[0]
# Récupération de la connectivité donnant pour chaque triangle l'indice
# de ses sommets. Tableau de dimension (nbre triangles, 3)
elt2verts = m[1]
nb_elts   = elt2verts.shape[0]

# A partir de elt2verts, on calcul la connectivité inverse, c'est à dire
# pour le sommet s, quels sont les indices des triangles contenant ce sommet
# s ? Puisque le nombre de triangle contenant un sommet diffère selon le
# sommet, on adopte une structure "creuse", c'est à dire un premier tableau,
# begVert2Elts,
# donnant pour chaque sommet l'indice dans le deuxième tableau
# vert2elts, où commence
# les indices des triangles le contenant. La dernière valeurs de ce premier
# tableau contient le nombre total d'indices contenus dans le deuxième
# tableau.
# Les indices de tous les triangles contenant un sommet s sont contenus dans
# le sous-tableau
#            vert2elts[begVert2Elts[s]:begVert2Elts[s+1]]
begVert2Elts, vert2elts = mesh.compvert2elts(elt2verts)

# A partir des connectivités triangles vers sommets et sommets vers triangles,
# on peut calculer le graphe de la matrice creuse qui sera stockée sous la forme
# CSC ( Compress Store Column ) consistant à donner dans un premier tableau
# l'indice dans les deux autres tableaux du début des indices lignes et des
# coefficients non nuls de la matrice.
# Dans le deuxième tableau, on récupère les indices lignes des éléments
# non nuls.
begCols, indRows = fem.comp_skel_csc_mat(elt2verts, (begVert2Elts, vert2elts) )
# le dernier élément de begCols contient le nombre d'éléments non nuls
# de la matrice :
nnz = begCols[-1]
print "Nombre d'éléments non nuls de la matrice : %04d\n"%nnz

# Connaissant le nombre d'éléments non nuls de la matrice et les indices
# lignes non nuls de la matrice, on peut assembler la matrice creuse
# sans tenir compte des conditions limites dans un premier temps.
spCoefs = np.zeros( nnz, np.double )
# On alloue également un tableau pour tenir compte de la matrice de masse
# diagonalisée
lump_matrix = np.zeros( nb_verts, np.double )

# Assemblage de la matrice creuse ( sans les conditions limites ) :
# On parcourt chaque élément ( triangle ) du maillage :
for iElt in xrange(nb_elts) :
    # On récupère les indices des sommets du triangle
    indSommets = elt2verts[iElt,:]
    # Puis les coordonnées de ces sommets :
    crd1 = coords[indSommets[0],:]
    crd2 = coords[indSommets[1],:]
    crd3 = coords[indSommets[2],:]
    # A partir de ces coordonnées ( dont on ignore le z toujours égal à zéro ! )
    # on calcule la matrice élémentaire du laplacien sur ce triangle :
    # integral_{triangle courant} grad(phi_i)grad(phi_j) dT
    matElem = laplacian.comp_eltmat((crd1[0],crd1[1]), (crd2[0],crd2[1]),
                                    (crd3[0],crd3[1]))
    # On rajoute la contribution de cette matrice élémentaire à la matrice
    # creuse
    fem.add_elt_mat_to_csc_mat( (begCols, indRows, spCoefs),
                                (indSommets, indSommets, matElem) )
    # On en profite en même temps pour assembler la matrice de masse
    # diagonalisée
    diff = crd2[0:2] - crd1[0:2]
    lgth = sqrt(diff.dot(diff))
    if crd1[3] > 0 :
        lump_matrix[indSommets[0]] += 0.5*lgth
    if crd2[3] > 0 :
        lump_matrix[indSommets[1]] += 0.5*lgth
            
    diff = crd3[0:2] - crd1[0:2]
    lgth = sqrt(diff.dot(diff))
    if crd1[3] > 0 :
        lump_matrix[indSommets[0]] += 0.5*lgth
    if crd3[3] > 0 :
        lump_matrix[indSommets[2]] += 0.5*lgth

    diff = crd3[0:2] - crd2[0:2]
    lgth = sqrt(diff.dot(diff))
    if crd2[3] > 0 :
        lump_matrix[indSommets[1]] += 0.5*lgth
    if crd3[3] > 0 :
        lump_matrix[indSommets[2]] += 0.5*lgth

# On assemble le second membre :
#   1. D'abord la fonction donnant les valeurs de la solution sur les sommets
#      correspondant à une condition de Dirichlet.
f = np.zeros(nb_verts, np.double)
for iVert in xrange(nb_verts) :
    if ( coords[iVert,3] > 0) :
        f[iVert] += g(coords[iVert,0], coords[iVert,1])
# puis la "contribution de ce f par rapport aux coefficients de la matrice
# assemblée qui correspond aux interactions des noeuds voisins avec
# les noeuds sur les conditions limites de Dirichlet :
b = np.zeros(nb_verts, np.double)
for j in xrange(nb_verts) :
    for ptR in xrange(begCols[j],begCols[j+1]):
        b[indRows[ptR]] -= spCoefs[ptR]*f[j]
# Puis on reimpose les conditions limites pour b :
for iVert in xrange(nb_verts) :
    if ( coords[iVert,3] > 0) :
        b[iVert] += f[iVert]

# Il faut maintenant tenir compte des conditions limites pour les coefficients
# de la matrice :
for iVert in xrange(nb_verts):
    if coords[iVert,3] > 0: # C'est une condition limite !
        # annulation de la colonne iVert avec 1 sur la diagonale :
        for i in xrange(begCols[iVert],begCols[iVert+1]):
            if indRows[i] != iVert :
                spCoefs[i] = 0.
            else :
                spCoefs[i] = 1.
        # Suppression des coefficients se trouvant sur la ligne iVert
        # ( avec toujours 1 sur la diagonale )
        for iCol in xrange(nb_verts):
            if iCol != iVert :
                for ptRow in xrange(begCols[iCol],begCols[iCol+1]):
                    if indRows[ptRow] == iVert :
                        spCoefs[ptRow] = 0.
                        
# On definit ensuite la matrice comme matrice CSC :
spMatrix = sparse.csc_matrix((spCoefs, indRows, begCols))

# Visualisation second membre :
VS.view( coords, elt2verts, b, title = "second membre", visuMesh = False )

# Factorisation creuse de la matrice 
AFact = sp_linalg.splu(spMatrix)
# On regarde le nombre d'éléments non nuls après factorisation :
print "nnz(AFact) : ", AFact.nnz

# Résolution du problème linéaire issue de la discrétisation par
# éléments finis :
sol = AFact.solve(b)

# Calcul de l'erreur relative faite avec la résolution directe :
d = spMatrix.dot(sol) - b # Calcul de A * sol - b
print "||A.x-b||/||b|| = ", sqrt(d.dot(d)/b.dot(b))

# Visualisation de la solution :
VS.view( coords, elt2verts, sol, title = "Solution", visuMesh = False )
