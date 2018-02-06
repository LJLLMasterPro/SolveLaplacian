#!/bin/env python
# -*- coding: utf-8 -*-
"""
Gradient conjugué éventuellement préconditionné
"""
def solve( A, x0, b, nb_iter_max, epsilon, Prec = None ) :
    """
    Résoud un système linéaire par gradient conjugué.
    Le gradient conjugué peut être conjugué si on passe
    un préconditionneur via Prec. Le préconditionneur doit
    posséder dans ce cas une méthode solve.
    Les arguments d'entrée sont :
      A : La matrice Lhs
      x0: La solution initiale
      b : Le second membre
      nb_iter_max : Le nombre maximal d'itérations avant que le gc s'arrête.
      epsilon : L'erreur relative faite sur la solution -> ||rk||/||b||avec rk = b-A.xk
      Prec ( optionnel ) : Le préconditionneur
    """
    rk = b - A.dot(x0)
    error = rk.dot(rk)
    error_init = b.dot(b)
    if (Prec is None):
        pk = rk
    else:
        pk = Prec.solve(rk)
    zk = pk
    xk = x0.copy()
    iter = 0
    while ( iter < nb_iter_max ) and ( error > epsilon*epsilon*error_init):
        Apk = A.dot(pk)
        rkdzk = rk.dot(zk)
        alpha = rkdzk/Apk.dot(pk)
        xk = xk + alpha * pk
        rk = rk - alpha * Apk
        error = rk.dot(rk)
        if (Prec is None) :
            zk = rk
        else:
            zk = Prec.solve(rk)
        betak = rk.dot(zk)/rkdzk
        pk = zk + betak*pk
        iter += 1
    return xk, iter, error
    
