from GSL cimport (gsl_function,
                   gsl_integration_cquad,
                   gsl_integration_cquad_workspace,
                   gsl_integration_cquad_workspace_alloc,
                   gsl_integration_cquad_workspace_free)

cdef double integrate_cython(double lower_lim, 
                             double upper_lim, 
                             double epsabs) nogil
