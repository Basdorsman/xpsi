#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport exp, pow
from libc.stdio cimport printf, fflush, stdout

from xpsi.global_imports import _c_cgs, _h_keV

cdef double c_cgs = _c_cgs
cdef double h_keV = _h_keV


cdef double _b_E(double T, 
                double E) nogil:
    '''
    Spectral radiance type thing of a blackbody.

    parameters:
        E in keV
        T in keV

    returns:
        b_E in photons/s/keV/cm^2/sr 
    '''
    return 2. * E * E / (h_keV * h_keV * h_keV * c_cgs * c_cgs) / (exp(E / T) - 1)

def b_E(double T,
        double E):
    
    return _b_E(T, E)

cdef double _disk_integrand(double T, 
                             void *params) nogil:
    
        '''
        Special spectral radiance to be integrated for the multicolor disk.
        
        parameters:
            T, T_in in keV
            E in keV

        returns:
            integrand in spectral radiance units/keV (you will integrate over T in keV)
        '''

        cdef double E = (<double**> params)[0][0]
        cdef double T_in = (<double**> params)[1][0]
        # printf('T: %f\n', T)
        # printf('E: %f\n', E)
        # printf('T_in: %f\n', T_in)
        # printf('(T/T_in)**(-11/3): %f\n', pow((T/T_in), (-11./3.)))
        # printf('spectral_radiance(E, T)/T_in: %f\n', _b_E(T, E)/T_in)
        # fflush(stdout)

        integrand = pow((T/T_in), (-11./3.))*_b_E(T, E)/T_in
        return integrand
    
def disk_integrand(double T,
             double E,
             double T_in):

    cdef void *params[2]
    params[0] = &E
    params[1] = &T_in

    return _disk_integrand(T, params)


cdef double _disk_f_E(double E, 
                     void *params) nogil:
    """
    For a given E, integrates the spectral radiance integrand from T_out to
    T_in using gsl_integration_cquad.

    Args:
        Energy: Energy of the photon
        lower_lim: Lower integration limit.
        upper_lim: Upper integration limit.
        epsrel: Desired absolute error.

    Returns:
        flux per energy of a multicolor disk
    """
    
    # unpacking parameters
    cdef double T_in = (<double**> params)[0][0]
    cdef double T_out = (<double**> params)[1][0]
    cdef double epsrel = (<double**> params)[2][0]

    # params for the integration of the integrand 
    cdef void *params_integrand[2]
    params_integrand[0] = &E
    params_integrand[1] = &T_in

    
    # Define GSL function structure
    cdef gsl_function F
    F.function = &_disk_integrand
    F.params = &params_integrand

    # Workspace allocation
    cdef gsl_integration_cquad_workspace *w_disk_f_E = gsl_integration_cquad_workspace_alloc(100)

    # Result and error variables
    cdef double integral, error
    cdef size_t nevals

    # Perform integration
    cdef int status = gsl_integration_cquad(&F, T_out, T_in, 0.0, epsrel, w_disk_f_E, &integral, &error, &nevals)
    # cdef int status = gsl_integration_cquad(&F, T_out, T_in, 0.0, epsrel, work_pointer, &integral, &error, &nevals)

    # Free workspace
    gsl_integration_cquad_workspace_free(w_disk_f_E)

    # Return results
    return integral #, error, nevals
    

def disk_f_E(double E,
             double T_in,
             double T_out,
             double epsrel):

    cdef void *params[3]
    params[0] = &T_in
    params[1] = &T_out
    params[2] = &epsrel

    return _disk_f_E(E, params)

cdef double _disk_f(double E_lower, 
                   double E_upper,
                   double T_out,
                   double T_in,
                   double epsrel) nogil:
    """
    Integrates the flux per energy within some energy band [E_lower, E_upper].

    Args:
        E_lower: Lower integration limit.
        E_upper: Upper integration limit.
        T_out: outer temperature of the disk
        T_in: inner temperature of the disk.
        epsrel: Desired absolute error.

    Returns:
        Flux of a multicolor disk within the energy band
    """
    
    # printf('E_lower: %f\n', E_lower)
    # printf('E_Upper: %f\n',E_upper)
    # printf('T_out: %f\n',T_out)
    # printf('T_in: %f\n',T_in)
    # printf('epsrel: %f\n', epsrel)
    
    cdef void *params[4]
    params[0] = &T_in
    params[1] = &T_out
    params[2] = &epsrel

    # Define GSL function structure
    cdef gsl_function F
    F.function = &_disk_f_E
    F.params = &params

    # Workspace allocation
    cdef gsl_cq_work *w_disk_f = gsl_integration_cquad_workspace_alloc(100)
    
    # Result and error variables
    cdef double integral, error
    cdef size_t nevals

    # Perform integration
    cdef int status = gsl_integration_cquad(&F, E_lower, E_upper, 0.0, epsrel, w_disk_f, &integral, &error, &nevals)

    # printf('value: %f\n', integral)

    # Free workspace
    gsl_integration_cquad_workspace_free(w_disk_f)



    # Return results    
    return integral

def disk_f(double E_lower, 
           double E_upper,
           double T_out,
           double T_in,
           double epsrel):
    

    fflush(stdout) 


    
    value = _disk_f(E_lower, E_upper, T_in, T_out, epsrel) 
    return value