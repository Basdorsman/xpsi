from libc.stdlib cimport malloc, free

cdef double* init_local_variables(double g, double T) nogil

#cdef int free_local_variables(const double *const vec) nogil
