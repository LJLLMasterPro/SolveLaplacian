#!/usr/bin/env python

def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    #
    config = Configuration('Project', parent_package, top_path)

    #blas_info = get_info('mkl',0)
    #if not blas_info :
    #    blas_info = get_info('lapack_opt',0)
    #    if not blas_info :
    #        print "### Warning no lapack/blas defined !"

    config.add_extension('fem',
                         sources=['CPyFem.cpp'])

    config.add_extension('laplacian',
                         sources=['CPyLaplacian.cpp'])

    config.add_extension('mesh',
                         sources=['CPyMesh.cpp'])

    config.add_extension('splitter',
                         sources = ['CPySplitter.cpp'],
                         extra_compile_args=['-std=c++11', '-pedantic', '-Wall', '-g'],
		                 include_dirs=['../include'],
                         library_dirs=['../lib'],
                         libraries=['metis'])

    #config.add_extension('AcaOp',
    #                     sources=['pyAcaOp.cpp'],
    #                     #extra_compile_args=['-g -O0'],
    #                     extra_info = blas_info)
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
