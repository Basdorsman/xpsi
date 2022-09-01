#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: embedsignature=True

from preload cimport (_preloaded, init_preload, free_preload)
from hot cimport (init_hot, eval_hot, eval_hot_norm, free_hot)
# from local_variables cimport init_local_variables
from libc.stdio cimport printf
# from libc.stdlib cimport malloc, free

def interpolate(size_t N_T,
                double E_prime,
                double cos_zenith,
                double tau,
                double t_bb,
                double t_e,
                atmosphere = None):

    cdef _preloaded *preloaded = NULL
    cdef void *data = NULL

    # cdef double *vec = NULL
    # vec = init_local_variables(g, T)

    if atmosphere:
        preloaded = init_preload(atmosphere)
        data = init_hot(N_T, preloaded)
    else:
        data = init_hot(N_T, NULL)
    
    
    cdef:
        size_t Thread
        double I_E

    # printf("input parameters reporting:")
    # printf("\nvec[0]: %.8e, ", vec[0])
    # printf("\nvec[1]: %.8e, ", vec[1])
    # printf("\ng: %.8e, ", g)
    # printf("\nT: %.8e, ", T)
    # printf("\nE_prime: %.8e, ", E_prime)
    # printf("\ncos_zenith: %.8e, ", cos_zenith)

    for Thread in range(N_T):
        I_E = eval_hot(Thread,
                    E_prime,
                    cos_zenith,
                    tau,
                    t_bb,
                    t_e,
                    data)
    
    free_preload(preloaded)
    free_hot(N_T, data)

    return I_E

