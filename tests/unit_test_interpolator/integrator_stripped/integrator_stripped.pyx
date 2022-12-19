#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: embedsignature=True

from preload cimport (_preloaded, init_preload, free_preload)
from hot cimport (init_hot, eval_hot, eval_hot_norm, free_hot, multiply_pointers, multiply_vectors)
# from local_variables cimport init_local_variables
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

# cdef extern from "<valarray>" namespace "std":
#     cdef cppclass valarray[T]:
#         valarray()
#         valarray(int)  # constructor: empty constructor
#         T& operator[](int)  # get/set element



def interpolate(size_t N_T,
                double E_prime,
                double cos_zenith,
                double tau,
                double t_bb,
                double t_e,
                atmosphere = None):

    #printf("inside interpolate()")
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

    # # vector stuff
    # cdef vector[int] vect
    # cdef int i, x
    
    # # populate vector with content
    # for i in range(10):
    #     vect.push_back(i)
        
    
    # for i in range(10):
    #     print(vect[i])
    
    # for x in vect:
    #     print(x)
    
    cdef int length=1000
        
    
    cdef vector[double] vecx
    cdef vector[double] vecy
    cdef vector[double] vecxy
    
    vecx = vector[double](length)
    vecy = vector[double](length)
    vecxy = vector[double](length)
        
    for i in range(length):
        vecx[i]=i
        vecy[i]=i


    print("testing vector multiplication speed")
    
    cdef:
        timespec ts, te
        long int t_forloop, ts_function, tns_function_here, t_function_here
    
    # clock_gettime(CLOCK_REALTIME, &ts)
    # for i in range(length):
    #     vecxy[i] = vecx[i] * vecy[i]
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # t_forloop = (te.tv_nsec - ts.tv_nsec)
    # printf("multiplication in forloop takes %ld ns\n",t_forloop)
    
    
    # clock_gettime(CLOCK_REALTIME, &ts)
    #multiply_vectors(length, vecxy, vecx, vecy)
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # t_function = (te.tv_nsec - ts.tv_nsec)
    # printf("multiplication in function takes %ld ns\n",t_function)
    
    # clock_gettime(CLOCK_REALTIME, &ts)
    vecxy = multiply_vectors_here(vecxy, vecx, vecy, length)
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # tns_function_here = (te.tv_nsec - ts.tv_nsec)
    # ts_function_here = (te.tv_sec - ts.tv_sec)
    # printf("multiplication in function here takes %ld s %ld ns\n",ts_function_here,tns_function_here)
    
    # for i in range(length):
    #     print(vecxy[i])


    # vector multiplication with pointers
    printf("vector multiplication with pointers:\n")
    
    cdef long int t_pointers, t_pointers_function
    cdef timespec tps, tpe
    
    cdef double* vec_point_xy = <double*>malloc(length * sizeof(double))
    cdef double* vec_point_x = <double*>malloc(length * sizeof(double))
    cdef double* vec_point_y = <double*>malloc(length * sizeof(double))
    
    for i in range(length):
        vec_point_x[i]=i
        vec_point_x[i]=i
    
    clock_gettime(CLOCK_REALTIME, &tps)
    for i in range(length):
        vec_point_xy[i] = vec_point_x[i] * vec_point_y[i]
    clock_gettime(CLOCK_REALTIME, &tpe)
    
    t_pointers = (tpe.tv_nsec - tps.tv_nsec)
    printf("multiplication in pointers takes %ld ns\n",t_pointers)
    
    clock_gettime(CLOCK_REALTIME, &tps)
    vec_point_xy = multiply_pointers(vec_point_xy, vec_point_x, vec_point_y, length)
    clock_gettime(CLOCK_REALTIME, &tpe)
    
    t_pointers_function = (tpe.tv_nsec - tps.tv_nsec)
    printf("multiplication in pointers in function takes %ld ns\n",t_pointers_function)
    
    free(vec_point_x)
    free(vec_point_y)
    free(vec_point_xy)
    
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

def multiply_vectors_here(vecxy, vecx, vecy, length):
    for i in range(length):
        vecxy[i] = vecx[i] * vecy[i]
    return vecxy

