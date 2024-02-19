#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cdef double my_function(double x, void* params) nogil:
    return x

cdef double integrate_cython(double lower_lim, 
                             double upper_lim, 
                             double epsabs) nogil:
    """
    Integrates the function y = x from lower_lim to upper_lim using gsl_integration_cquad.

    Args:
        lower_lim: Lower integration limit.
        upper_lim: Upper integration limit.
        epsabs: Desired absolute error.

    Returns:
        A tuple containing the integral value, error, and number of function evaluations.
    """

    cdef double params[3]

    # Define GSL function structure
    cdef gsl_function F
    F.function = my_function
    F.params = &params

    # Workspace allocation
    cdef gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(100)

    # Result and error variables
    cdef double integral, error
    cdef size_t nevals

    # Perform integration
    cdef int status = gsl_integration_cquad(&F, lower_lim, upper_lim, epsabs, 0.0, w, &integral, &error, &nevals)


    # Free workspace
    gsl_integration_cquad_workspace_free(w)

    # Return results
    return integral #, error, nevals


def integrate(double lower_lim,
              double upper_lim,
              double epsabs):
    
    integral = integrate_cython(lower_lim, 
                                upper_lim, 
                                epsabs)
    
    return integral
    
    
    
    
    
# Example usage
#integral, error, nevals = integrate_cython(0.0, 1.0, 1.0e-8)
#print("Integral:", integral)
#print("Error:", error)
#print("Number of evaluations:", nevals)