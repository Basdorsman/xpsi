#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

cdef int SUCCESS = 0

cdef double* init_local_variables(double g, double T) nogil:
    vec = <double*> malloc(2 * sizeof(double))
    vec[0] = g
    vec[1] = T
    return vec

# cdef int free_local_variables(const double *const vec) nogil:
#     free(vec)
#     return SUCCESS