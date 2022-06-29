#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cdef _preloaded* init_preload(object atmosphere):
    cdef size_t i, j
    cdef double[::1] cast
    cdef double[::1] intensity

    preloaded = <_preloaded*> malloc(sizeof(_preloaded))

    preloaded.ndims = <size_t>(len(atmosphere) - 1)

    preloaded.params = <double**> malloc(sizeof(double*) * preloaded.ndims)
    preloaded.N = <size_t*> malloc(sizeof(size_t) * (preloaded.ndims))
    preloaded.S = <size_t*> malloc(sizeof(size_t) * (preloaded.ndims - 1))
    preloaded.BLOCKS = <size_t*> malloc(sizeof(size_t) * (preloaded.ndims - 1))
    
    # print("memory allocated.")
    # print("preloaded.S[0]:")
    # print(preloaded.S[0])
    # print("preloaded.N[0]:")
    # print(preloaded.N[0])
    
    
    for i in range(preloaded.ndims):
        cast = atmosphere[i]
        #print("preload, i={}".format(i))
        preloaded.N[i] = cast.shape[0]
        #print("preloaded.N[i]: {}".format(preloaded.N[i]))
        #print("preloaded.S[i]: {}".format(preloaded.S[i]))
        preloaded.params[i] = &cast[0]
        if i < preloaded.ndims - 1:
            #print("do if statement to make S")
            cast = atmosphere[i+1]
            preloaded.S[i] = cast.shape[0]
            #print("preloaded.S[i]: {}".format(preloaded.S[i]))
            if i < preloaded.ndims - 2:
                #print("second if statement")
                for j in range(i+2, preloaded.ndims):
                    cast = atmosphere[j]
                    #print("multiply by jth shape: {}".format(cast.shape[0]))
                    preloaded.S[i] *= cast.shape[0]
                    #print("preloaded.S[i]: {}".format(preloaded.S[i]))
        #print("final result")
        #print("preloaded.S[i]: {}".format(preloaded.S[i]))
    # print("calculate intensity")
    # print("i: {}".format(i))
    # print("atmosphere[i]: {}".format(atmosphere[i]))
    # print("atmosphere[i+1]: {}".format(atmosphere[i+1]))
    intensity = atmosphere[i+1]
    # print("intensity: {}".format(intensity))
    # print("intensity[0]: ".format(intensity[0]))
    preloaded.I = &intensity[0]
    # print("preloaded.I {}".format(preloaded.I))
    return preloaded

cdef int free_preload(_preloaded *const preloaded) nogil:
    free(preloaded.params)
    free(preloaded.N)
    free(preloaded.S)
    free(preloaded.BLOCKS)
    free(preloaded)

    return 0
