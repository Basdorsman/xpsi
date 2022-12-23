#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: embedsignature=True
from __future__ import print_function
from preload cimport (_preloaded, init_preload, free_preload)
from hot cimport (init_hot, eval_hot, eval_hot_norm, free_hot, eval_hot_seploops) #simd_add)#, multiply_pointers, add_pointers)#, multiply_vectors)
# from local_variables cimport init_local_variables
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

# cimport numpy as np 

# cdef extern from "cpuid.h":
#     cdef bool support
#     # __builtin_cpu_init()
#     if __builtin_cpu_supports():
#         support = True
#     else:
#         support = False

# print(support)

# cdef extern from *:
#     """
#     int cpu_supports_avx(void){
#         return __builtin_cpu_supports("avx");
#     }
#     """
#     int cpu_supports_avx()

# def cpu_has_avx_support():
#     return cpu_supports_avx() != 0


# print('supports avx:', cpu_has_avx_support())

 

# cdef extern from "<valarray>" namespace "std":
#     cdef cppclass valarray[T]:
#         valarray()
#         valarray(int)  # constructor: empty constructor
#         T& operator[](int)  # get/set element

# cdef extern from "emmintrin.h":  # in this example, we use SSE2
#     ctypedef double __m128d
#     __m128d _mm_loadu_pd (double *__P) nogil  # (__P[0], __P[1]) are the original pair of doubles
#     __m128d _mm_add_pd (__m128d __A, __m128d __B) nogil
#     __m128d _mm_mul_pd (__m128d __A, __m128d __B) nogil
#     void _mm_store_pd (double *__P, __m128d __A) nogil  # result written to (__P[0], __P[1])

# cdef void example1():
#     cdef double[2] data
#     cdef __m128d mdata, mtmp
#     cdef double[2] out


#     data[:] = [1.0, 3.0]

#     with nogil:
#         # pack double pairs into __m128d
#         mdata = _mm_loadu_pd( &data[0] )

#         # add:  tmp = data + data
#         mtmp = _mm_add_pd( mdata, mdata )

#         # unpack result
#         _mm_store_pd( &out[0], mtmp )

#     print( " in", data[0], data[1], sep=", " )  # "1.0, 3.0"
#     print( "out", out[0],  out[1],  sep=", " )  # "2.0, 6.0"


# cdef extern from "immintrin.h":
#     ctypedef float __m128
#     __m128 _mm_load_ps(float*__P) nogil
#     __m128 _mm_add_ps(__m128 __A, __m128 __B) nogil
#     void _mm_store_ps(float* __P, __m128 __A) nogil

# cdef void simd_add(float[:] a, float[:] b, float[:] c):
#     cdef __m128 ma, mb, mtemp
#     cdef int i
#     with nogil:
#         for i in range(0, len(a), 4):
#             ma = _mm_load_ps(&a[i])
#             mb = _mm_load_ps(&b[i]) 
#             mtemp = _mm_add_ps(ma, mb)
#             _mm_store_ps(&c[i], mtemp)
    

# ctypedef struct mystructure:
#     double* temp2
#     double* I_CACHE
#     double* I_temp

from hot cimport mystruct

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


    #################################################3 vector stuff
    # cdef vector[int] vect
    # cdef int i, x
    
    # # populate vector with content
    # for i in range(10):
    #     vect.push_back(i)
        
    
    # # for i in range(10):
    # #     print(vect[i])
    
    # # for x in vect:
    # #     print(x)
    
    # cdef int length=1000
        
    
    # cdef vector[double] vecx
    # cdef vector[double] vecy
    # cdef vector[double] vecxy
    
    # vecx = vector[double](length)
    # vecy = vector[double](length)
    # vecxy = vector[double](length)
        
    # for i in range(length):
    #     vecx[i]=i
    #     vecy[i]=i


    # print("testing vector multiplication speed")
    # cdef int repeats = 10
    
    # cdef:
    #     timespec ts, te
    #     long int t_forloop, ts_function, tns_function_here, t_function_here
    
    # clock_gettime(CLOCK_REALTIME, &ts)
    # for rep in range(repeats):
    #     for i in range(length):
    #         vecxy[i] = vecx[i] * vecy[i]
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # t_forloop = (te.tv_nsec - ts.tv_nsec) / repeats
    # printf("multiplication vectors in forloop takes %ld ns\n",t_forloop)
    
    
    # clock_gettime(CLOCK_REALTIME, &ts)
    # multiply_vectors(length, vecxy, vecx, vecy)
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # t_function = (te.tv_nsec - ts.tv_nsec)
    # printf("multiplication in function takes %ld ns\n",t_function)
    
    # clock_gettime(CLOCK_REALTIME, &ts)
    # vecxy = multiply_vectors_here(vecxy, vecx, vecy, length)
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # tns_function_here = (te.tv_nsec - ts.tv_nsec)
    # ts_function_here = (te.tv_sec - ts.tv_sec)
    # printf("multiplication in function here takes %ld s %ld ns\n",ts_function_here,tns_function_here)
    
    # for i in range(length):
    #     print(vecxy[i])


    ############################# vector multiplication with pointers
    # printf("vector multiplication with pointers:\n")
    
    # cdef long int t_pointers, t_pointers_function
    # cdef timespec tps, tpe
    
    # cdef double* vec_point_xy = <double*>malloc(length * sizeof(double))
    # cdef double* vec_point_x = <double*>malloc(length * sizeof(double))
    # cdef double* vec_point_y = <double*>malloc(length * sizeof(double))
    
    # for i in range(length):
    #     vec_point_x[i]=i
    #     vec_point_x[i]=i
    
    # clock_gettime(CLOCK_REALTIME, &tps)
    # for rep in range(repeats):
    #     for i in range(length):
    #         vec_point_xy[i] = vec_point_x[i] * vec_point_y[i]
    # clock_gettime(CLOCK_REALTIME, &tpe)
    
    # t_pointers = (tpe.tv_nsec - tps.tv_nsec) / repeats
    # t_s_pointers = (tpe.tv_sec - tps.tv_sec) / repeats
    # printf("multiplication in pointers takes %ld ns\n",t_pointers)
    # printf("multiplication in pointers takes %ld s\n",t_s_pointers)
    
    # clock_gettime(CLOCK_REALTIME, &tps)
    # for rep in range(repeats):
    #     vec_point_xy = multiply_pointers(vec_point_xy, vec_point_x, vec_point_y, length)
    # clock_gettime(CLOCK_REALTIME, &tpe)
    
    # t_pointers_function = (tpe.tv_nsec - tps.tv_nsec) / repeats
    # printf("multiplication in pointers in function takes %ld ns\n",t_pointers_function)
    
    
    # clock_gettime(CLOCK_REALTIME, &tps)
    # for rep in range(repeats):
    #     vec_point_xy = multiply_pointers(vec_point_xy, vec_point_x, vec_point_y, length)
    # clock_gettime(CLOCK_REALTIME, &tpe)
    
    # t_pointers_function = (tpe.tv_nsec - tps.tv_nsec) / repeats
    # printf("addition in pointers in function takes %ld ns\n",t_pointers_function)
    
        
    # free(vec_point_x)
    # free(vec_point_y)
    # free(vec_point_xy)

    ######## SIMD stuff
    # import numpy as np

    # # Create some random arrays of floats using NumPy
    # a = np.random.random(1000).astype(np.float32)
    # b = np.random.random(1000).astype(np.float32)
    
    # # Create an array to store the result of the addition
    # c = np.empty_like(a)
    
    # # Call the simd_add function to perform the SIMD addition
    # simd_add(a, b, c)

    # example1()
    

    # cdef float[1000] a, b, c
    
    # for i in range(length):
    #     a[i]=i
    #     b[i]=i
    
    # clock_gettime(CLOCK_REALTIME, &ts)
    # for rep in range(repeats):
    #     simd_add(a, b, c)
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # ts_function = (te.tv_nsec - ts.tv_nsec)/repeats
    # printf("addition in quads takes %ld ns\n",ts_function)

    # clock_gettime(CLOCK_REALTIME, &ts)
    # for rep in range(repeats):
    #     for i in range(length):
    #         c[i] = a[i] + b[i]
    # clock_gettime(CLOCK_REALTIME, &te)
    
    # ts_function = (te.tv_nsec - ts.tv_nsec)/repeats
    # printf("addition with scalars takes %ld ns\n",ts_function)
    
    cdef timespec ts, te
    cdef size_t t_elapsed
    cdef int iterator
    cdef int iteration_size = 1024
    cdef double* temp2 #= <double*>malloc(iteration_size * sizeof(double))
    # cdef double* I_temp = <double*>malloc(iteration_size * sizeof(double))
    cdef double* I_CACHE #= <double*>malloc(iteration_size * sizeof(double))
    cdef float[1024] temp2_store, I_CACHE_store
    cdef mystruct temps

    clock_gettime(CLOCK_REALTIME, &ts)
    printf('here is eval_hot().\n')

    for Thread in range(N_T):
        I_E = eval_hot(Thread,
                    E_prime,
                    cos_zenith,
                    tau,
                    t_bb,
                    t_e,
                    data)

    clock_gettime(CLOCK_REALTIME, &te)
    t_elapsed = (te.tv_nsec - ts.tv_nsec)
    printf("eval_hot() old takes %ld ns\n",t_elapsed) 

    
    clock_gettime(CLOCK_REALTIME, &ts)
    printf('here is new eval_hot().')
    # printf('\nbefore eval_hot_seploops: %lu',ts.tv_nsec)
        
    for Thread in range(N_T):
        temps = eval_hot_seploops(Thread,
                                 E_prime,
                                 cos_zenith,
                                 tau,
                                 t_bb,
                                 t_e,
                                 data)
        
    clock_gettime(CLOCK_REALTIME, &te)
    # printf('\nafter eval_hot_seploops: %lu',te.tv_nsec)
    t_elapsed = (te.tv_nsec - ts.tv_nsec)
    printf("\n new eval_hot() takes %ld ns\n",t_elapsed) 
    
    
    clock_gettime(CLOCK_REALTIME, &ts)
    printf('here is eval_hot() (2nd time).\n')

    for Thread in range(N_T):
        I_E = eval_hot(Thread,
                    E_prime,
                    cos_zenith,
                    tau,
                    t_bb,
                    t_e,
                    data)
        
    clock_gettime(CLOCK_REALTIME, &te)
    t_elapsed = (te.tv_nsec - ts.tv_nsec)
    printf("eval_hot() old takes %ld ns\n",t_elapsed) 
    
    free_preload(preloaded)
    free_hot(N_T, data)
    
        
    for iterator in range(iteration_size):
        temp2_store[iterator]=temps.temp2[iterator]
        I_CACHE_store[iterator]=temps.I_CACHE[iterator]
        # print(storage[iterator])

    return temp2_store, I_CACHE_store

def multiply_vectors_here(vecxy, vecx, vecy, length):
    for i in range(length):
        vecxy[i] = vecx[i] * vecy[i]
    return vecxy

