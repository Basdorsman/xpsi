#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:30:45 2022

@author: bas
"""

# import os
# from setuptools import setup, Extension


# cmdclass = {}

# try:
#     import Cython
#     print('Cython.__version__ == %s' % Cython.__version__)
#     from Cython.Distutils import build_ext
# except ImportError:
#     print('Cannot use Cython. Trying to build extension from C files...')
#     try:
#         from distutils.command import build_ext
#     except ImportError:
#         print('Cannot import build_ext from distutils...')
#         raise
#     else:
#         cmdclass['build_ext'] = build_ext
#         file_extension = '.c'
# else:
#     print('Using Cython to build extension from .pyx files...')
#     file_extension = '.pyx'
#     cmdclass['build_ext'] = build_ext



# def EXTENSION(modname):

#     pathname = modname.replace('.', os.path.sep)
#     file_extension = '.pyx'

#     return Extension(modname,
#                      [pathname + file_extension],
#                      include_dirs = [os.path.dirname(os.path.abspath(__file__))]
#                      )


import os

from setuptools import setup, Extension

if __name__ == '__main__':
    import numpy
    import sys
    OS = sys.platform
    import os
    from os.path import join

    if 'darwin' in OS or 'linux' in OS:
        print('Operating system: ' + OS)

        try:
            import subprocess as sub
        except ImportError:
            print('The subprocess module is required to locate the GSL library.')
            raise

        try:
            gsl_version = sub.check_output(['gsl-config','--version'])[:-1]
            gsl_prefix = sub.check_output(['gsl-config','--prefix'])[:-1]
        except Exception:
            print('GNU Scientific Library cannot be located.')
            raise
        else:
            print('GSL version: ' + gsl_version)
            libraries = ['gsl','gslcblas','m'] # default BLAS interface for gsl
            library_dirs = [gsl_prefix + '/lib']
            _src_dir = os.path.dirname(os.path.abspath(__file__))
            include_dirs = [gsl_prefix + '/include',
                            numpy.get_include(),
                            join(_src_dir, '')]

            # point to shared library at compile time so runtime resolution
            # is not affected by environment variables, but is determined
            # by the binary itself
            extra_link_args = ['-Wl,-rpath,%s'%(gsl_prefix+'/lib')]

        # try to get the rayXpanda library:
        # please modify these compilation steps it does not work for your
        # environment; this specification of the (shared) object files
        # seems to work fine for gcc and icc compilers at least
        # try:
        #     import rayXpanda
        # except ImportError:
        #     print('Warning: the rayXpanda package cannot be imported. '
        #           'Using fallback implementation.')
        #     CC = os.environ['CC']
        #     sub.call(['%s'%CC,
        #               '-c',
        #               join(_src_dir, 'xpsi/include/rayXpanda/inversion.c'),
        #               '-o',
        #               join(_src_dir, 'xpsi/include/rayXpanda/inversion.o')])
        #     sub.call(['%s'%CC,
        #               '-c',
        #               join(_src_dir, 'xpsi/include/rayXpanda/deflection.c'),
        #               '-o',
        #               join(_src_dir, 'xpsi/include/rayXpanda/deflection.o')])
        #     use_rayXpanda = False
        # else:
        #     use_rayXpanda = True

        # if use_rayXpanda:
        #     if 'clang' in os.environ['CC']:
        #         libraries += ['inversion.so', 'deflection.so']
        #     else:
        #         libraries += [':inversion.so', ':deflection.so']
        #     library_dirs += [rayXpanda.__path__[0]]
        #     extra_link_args += ['-Wl,-rpath,%s'%rayXpanda.__path__[0]]
        # else: # get the native dummy interface
        #     if 'clang' in os.environ['CC']:
        #         libraries += ['inversion.o', 'deflection.o']
        #     else:
        #         libraries += [':inversion.o', ':deflection.o']
        #     library_dirs += [join(_src_dir, 'xpsi/include/rayXpanda')]
        #     extra_link_args += ['-Wl,-rpath,%s'%join(_src_dir,
        #                                          'xpsi/include/rayXpanda')]

        try:
            if 'gcc' in os.environ['CC']:
                extra_compile_args=['-fopenmp',
                                    #'-fopt-info-vec-all', #vectorization info
                                    '-march=native',
                                    '-O3',
                                    '-funroll-loops',
                                    '-Wno-unused-function',
                                    '-Wno-uninitialized',
                                    '-Wno-cpp']
                extra_link_args.append('-fopenmp')
            elif 'icc' in os.environ['CC']:
                # on high-performance systems using Intel processors
                # on compute nodes, it is usually recommended to select the
                # instruction set (extensions) optimised for a given processor
                extra_compile_args=['-qopenmp',
                                    '-O3',
                                    '-xHOST',
                                    # alternative instruction set
                                    '-axCORE-AVX2,AVX',
                                    '-funroll-loops',
                                    '-Wno-unused-function']
                extra_link_args.append('-qopenmp')
            elif 'clang' in os.environ['CC']:
                extra_compile_args=['-fopenmp',
                                    '-Wno-unused-function',
                                    '-Wno-uninitialized',
                                    '-Wno-#warnings',
                                    '-Wno-error=format-security']
                extra_link_args.append('-fopenmp')
                # you might need these lookup paths for llvm clang on macOS
                # or you might need to edit these paths for your compiler
                #library_dirs.append('/usr/local/opt/llvm/lib')
                #include_dirs.append('/usr/local/opt/llvm/include')
        except KeyError:
            print('Export CC environment variable to "icc" or "gcc" or '
                  '"clang", or modify the setup script for your compiler.')
            raise
    else:
        print('Unsupported operating system. Manually inspect and modify '
              'setup.py script.')
        raise Exception

    cmdclass = {}

    try:
        import Cython
        print('Cython.__version__ == %s' % Cython.__version__)
        from Cython.Distutils import build_ext
    except ImportError:
        print('Cannot use Cython. Trying to build extension from C files...')
        try:
            from distutils.command import build_ext
        except ImportError:
            print('Cannot import build_ext from distutils...')
            raise
        else:
            cmdclass['build_ext'] = build_ext
            file_extension = '.c'
    else:
        print('Using Cython to build extension from .pyx files...')
        file_extension = '.pyx'
        cmdclass['build_ext'] = build_ext

    def EXTENSION(modname):

        pathname = modname.replace('.', os.path.sep)

        return Extension(modname,
                         [pathname + file_extension],
                         language = 'c++', #'c',
                         libraries = libraries,
                         library_dirs = library_dirs,
                         include_dirs = include_dirs,
                         extra_compile_args = extra_compile_args,
                         extra_link_args = extra_link_args)

modnames = ['integrator_stripped','hot','preload','local_variables']

extensions = []

for mod in modnames:
    extensions.append(EXTENSION(mod))

setup(
      name = 'integrator_stripped',
      ext_modules = extensions,
      cmdclass = cmdclass
)
