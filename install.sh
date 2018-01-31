#!/usr/bin/env bash
(cd thirdparty/metis-5.1.0; make config shared=1 prefix=`pwd`/../..; make; make install)
(cd Modules; python setup.py install --home=`pwd`/..)

