#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport M_PI, sqrt, sin, cos, acos, log10, pow, exp, fabs
from libc.stdio cimport printf, fopen, fclose, fread, FILE
from GSL cimport gsl_isnan, gsl_isinf
from libc.stdlib cimport malloc, free

from xpsi.global_imports import _keV, _k_B

cdef int SUCCESS = 0
cdef int ERROR = 1

cdef double erg = 1.0e-7
cdef double k_B = _k_B
cdef double keV = _keV
cdef double k_B_over_keV = k_B / keV
cdef int VERBOSE = 0

ctypedef struct ACCELERATE:
    size_t **BN                # base node for interpolation
    double **node_vals
    double **SPACE
    double **DIFF
    double **INTENSITY_CACHE
    double **VEC_CACHE

# Modify this struct if useful for the user-defined source radiation field.
# Note that the members of DATA will be shared by all threads and are
# statically allocated, whereas the members of ACCELERATE will point to
# dynamically allocated memory, not shared by threads.

ctypedef struct DATA:
    const _preloaded *p
    ACCELERATE acc

#----------------------------------------------------------------------->>>
# >>> User modifiable functions.
# >>> Note that the user is entirely free to wrap thread-safe and
# ... non-parallel external C routines from an external library.
# >>> Thus the bodies of the following need not be written explicitly in
# ... the Cython language.
#----------------------------------------------------------------------->>>
cdef void* init_hot(size_t numThreads, const _preloaded *const preloaded) nogil:
    # This function must match the free management routine free_hot()
    # in terms of freeing dynamically allocated memory. This is entirely
    # the user's responsibility to manage.
    # Return NULL if dynamic memory is not required for the model

    #printf("inside init_hot()")
    cdef DATA *D = <DATA*> malloc(sizeof(DATA))
    D.p = preloaded
    
    # (1) These BLOCKS appear to be related to the number of interpolation
    # points needed in a "hypercube". However, I would expect this to be 256 
    # for 4 dimensional interpolation already..

    # D.p.BLOCKS[0] = 64
    # D.p.BLOCKS[1] = 16
    # D.p.BLOCKS[2] = 4

    # By analogy, expand by one factor of four.

    D.p.BLOCKS[0] = 256    
    D.p.BLOCKS[1] = 64
    D.p.BLOCKS[2] = 16
    D.p.BLOCKS[3] = 4


    cdef size_t T, i, j, k, l, m

    D.acc.BN = <size_t**> malloc(numThreads * sizeof(size_t*))
    D.acc.node_vals = <double**> malloc(numThreads * sizeof(double*))
    D.acc.SPACE = <double**> malloc(numThreads * sizeof(double*))
    D.acc.DIFF = <double**> malloc(numThreads * sizeof(double*))
    D.acc.INTENSITY_CACHE = <double**> malloc(numThreads * sizeof(double*))
    D.acc.VEC_CACHE = <double**> malloc(numThreads * sizeof(double*))

    for T in range(numThreads):
        D.acc.BN[T] = <size_t*> malloc(D.p.ndims * sizeof(size_t))
        D.acc.node_vals[T] = <double*> malloc(2 * D.p.ndims * sizeof(double))
        D.acc.SPACE[T] = <double*> malloc(4 * D.p.ndims * sizeof(double))
        D.acc.DIFF[T] = <double*> malloc(4 * D.p.ndims * sizeof(double))
        #D.acc.INTENSITY_CACHE[T] = <double*> malloc(256 * sizeof(double))
        D.acc.INTENSITY_CACHE[T] = <double*> malloc(1024 * sizeof(double))
        D.acc.VEC_CACHE[T] = <double*> malloc(D.p.ndims * sizeof(double))
        for i in range(D.p.ndims):
            D.acc.BN[T][i] = 0
            D.acc.VEC_CACHE[T][i] = D.p.params[i][1]
            D.acc.node_vals[T][2*i] = D.p.params[i][1]
            D.acc.node_vals[T][2*i + 1] = D.p.params[i][2]

            j = 4*i

            D.acc.SPACE[T][j] = 1.0 / (D.p.params[i][0] - D.p.params[i][1])
            D.acc.SPACE[T][j] /= D.p.params[i][0] - D.p.params[i][2]
            D.acc.SPACE[T][j] /= D.p.params[i][0] - D.p.params[i][3]

            D.acc.SPACE[T][j + 1] = 1.0 / (D.p.params[i][1] - D.p.params[i][0])
            D.acc.SPACE[T][j + 1] /= D.p.params[i][1] - D.p.params[i][2]
            D.acc.SPACE[T][j + 1] /= D.p.params[i][1] - D.p.params[i][3]

            D.acc.SPACE[T][j + 2] = 1.0 / (D.p.params[i][2] - D.p.params[i][0])
            D.acc.SPACE[T][j + 2] /= D.p.params[i][2] - D.p.params[i][1]
            D.acc.SPACE[T][j + 2] /= D.p.params[i][2] - D.p.params[i][3]

            D.acc.SPACE[T][j + 3] = 1.0 / (D.p.params[i][3] - D.p.params[i][0])
            D.acc.SPACE[T][j + 3] /= D.p.params[i][3] - D.p.params[i][1]
            D.acc.SPACE[T][j + 3] /= D.p.params[i][3] - D.p.params[i][2]

            D.acc.DIFF[T][j] = D.acc.VEC_CACHE[T][i] - D.p.params[i][1]
            D.acc.DIFF[T][j] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][2]
            D.acc.DIFF[T][j] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][3]

            D.acc.DIFF[T][j + 1] = D.acc.VEC_CACHE[T][i] - D.p.params[i][0]
            D.acc.DIFF[T][j + 1] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][2]
            D.acc.DIFF[T][j + 1] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][3]

            D.acc.DIFF[T][j + 2] = D.acc.VEC_CACHE[T][i] - D.p.params[i][0]
            D.acc.DIFF[T][j + 2] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][1]
            D.acc.DIFF[T][j + 2] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][3]

            D.acc.DIFF[T][j + 3] = D.acc.VEC_CACHE[T][i] - D.p.params[i][0]
            D.acc.DIFF[T][j + 3] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][1]
            D.acc.DIFF[T][j + 3] *= D.acc.VEC_CACHE[T][i] - D.p.params[i][2]

    cdef double *address = NULL
    # Cache intensity
    # printf("\ncommencing cache intensity")
    
    # (2) For every dimension, we have a D.acc.BN[T][i], so I add a forloop to 
    # this. It pains me to have so many forloops.
    
    # for T in range(numThreads):
    #     for i in range(4):
    #         for j in range(4):
    #             for k in range(4):
    #                 for l in range(4):
    #                     address = D.p.I + (D.acc.BN[T][0] + i) * D.p.S[0]
    #                     address += (D.acc.BN[T][1] + j) * D.p.S[1]
    #                     address += (D.acc.BN[T][2] + k) * D.p.S[2]
    #                     address += D.acc.BN[T][3] + l
    #                     D.acc.INTENSITY_CACHE[T][i * D.p.BLOCKS[0] + j * D.p.BLOCKS[1] + k * D.p.BLOCKS[2] + l] = address[0]

    for T in range(numThreads):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        for m in range(4):
                            address = D.p.I + (D.acc.BN[T][0] + i) * D.p.S[0]
                            address += (D.acc.BN[T][1] + j) * D.p.S[1]
                            address += (D.acc.BN[T][2] + k) * D.p.S[2]
                            address += (D.acc.BN[T][3] + l) * D.p.S[3]
                            address += D.acc.BN[T][4] + m
                            D.acc.INTENSITY_CACHE[T][i * D.p.BLOCKS[0] + j * D.p.BLOCKS[1] + k * D.p.BLOCKS[2] + l * D.p.BLOCKS[3] + m] = address[0]


    # Cast for generalised usage in integration routines
    return <void*> D


cdef int free_hot(size_t numThreads, void *const data) nogil:
    # This function must match the initialisation routine init_hot()
    # in terms of freeing dynamically allocated memory. This is entirely
    # the user's responsibility to manage.
    # The void pointer must be appropriately cast before memory is freed --
    # only the user can know this at compile time.
    # Just use free(<void*> data) iff no memory was dynamically
    # allocated in the function:
    #   init_hot()
    # because data is expected to be NULL in this case

    # printf("inside free_hot()")

    cdef DATA *D = <DATA*> data

    cdef size_t T

    for T in range(numThreads):
        # printf("freeing thread specific memory")
        free(D.acc.BN[T])
        free(D.acc.node_vals[T])
        free(D.acc.SPACE[T])
        free(D.acc.DIFF[T])
        free(D.acc.INTENSITY_CACHE[T])
        free(D.acc.VEC_CACHE[T])

    # printf("freeing D.acc...")
    free(D.acc.BN)
    free(D.acc.node_vals)
    free(D.acc.SPACE)
    free(D.acc.DIFF)
    free(D.acc.INTENSITY_CACHE)
    free(D.acc.VEC_CACHE)

    # printf("freeing D...")
    free(D)

    return SUCCESS

#----------------------------------------------------------------------->>>
# >>> Cubic polynomial interpolation.
# >>> Improve acceleration properties... i.e. do not recompute numerical
# ... weights or re-read intensities if not necessary.
#----------------------------------------------------------------------->>>
cdef double eval_hot(size_t THREAD,
                     double E,
                     double mu,
                     const double *const VEC,
                     void *const data) nogil:

    cdef double g, Temperature
    Temperature = VEC[1]
    g = VEC[2]
    
    # Arguments:
    # E = photon energy in keV
    # mu = cosine of ray zenith angle (i.e., angle to surface normal)
    # VEC = variables such as temperature, effective gravity, ...
    # data = numerical model data required for intensity evaluation
    # This function must cast the void pointer appropriately for use.
    cdef DATA *D = <DATA*> data

    cdef:
        size_t i = 0, ii
        double I = 0.0, temp
        double *node_vals = D.acc.node_vals[THREAD]
        size_t *BN = D.acc.BN[THREAD]
        double *SPACE = D.acc.SPACE[THREAD]
        double *DIFF = D.acc.DIFF[THREAD]
        double *I_CACHE = D.acc.INTENSITY_CACHE[THREAD]
        double *V_CACHE = D.acc.VEC_CACHE[THREAD]
        double vec[5] # should be = ndims
        # double E_eff = k_B_over_keV * pow(10.0, VEC[0])
        double E_eff = k_B_over_keV * pow(10.0, Temperature)
        int update_baseNode[5]  # should be = ndims
        int CACHE = 0

    # (3) add my modulator variable. I don't know yet how to tunnel this from
    # the front end, so I won't. I am choosing value 5, which is in the middle of the range.
    # vec[0] = VEC[0]
    # vec[1] = VEC[1]
    # vec[2] = mu
    # vec[3] = log10(E / E_eff)
    
    

    vec[0] = 0 # THIS IS MY CUSTOM VARIABLE: MODULATES INTENSITY BY A FACTOR x10^-0.3-x10^0.3 #x1-10
    vec[1] = Temperature
    vec[2] = g
    vec[3] = mu
    vec[4] = log10(E / E_eff)
    
    
    # printf("diagnostics 0:\n")
    # printf("E: %.2e, ", E)
    # printf("Temperature: %.2e, ", Temperature)
    # printf("k_B_over_keV: %.2e, ", k_B_over_keV)
    # printf("E_eff: %.2e, ", E_eff)
    # printf("vec[4]: %.2e\n", vec[4])
    
    # printf("diagnostics 1:\n") # using i breaks next code block
    # for i in range(D.p.ndims):
    #     printf("i=%d, ", <int>i)
    #     printf("vec[i]: %.2e\n", vec[i])
    # cdef size_t test 

    # printf("\nvec[0]: %.8e, ", vec[0])
    # printf("vec[1]: %.8e, ", vec[1])
    # printf("vec[2]: %.8e, ", vec[2])
    # printf("vec[3]: %.8e, ", vec[3])
    # printf("vec[4]: %.8e, ", vec[4])
    
    #printf("\neval_hot() called")
    #printf("\nVEC[0]: %f", VEC[0])
    #printf("\nVEC[1]: %f", VEC[1])

    while i < D.p.ndims:
        # if parallel == 31:
        # printf("\nDimension: %d", <int>i)
        update_baseNode[i] = 0
        if vec[i] < node_vals[2*i] and BN[i] != 0:
            # if parallel == 31:
            # printf("\nExecute block 1: %d", <int>i)
            update_baseNode[i] = 1
            while vec[i] < D.p.params[i][BN[i] + 1]:
                # if parallel == 31:
                #     printf("\n!")
                #     printf("\nvec i: %.8e", vec[i])
                #     printf("\nBase node: %d", <int>BN[i])
                if BN[i] > 0:
                    BN[i] -= 1
                elif vec[i] <= D.p.params[i][0]:
                    vec[i] = D.p.params[i][0]
                    break
                elif BN[i] == 0:
                    break

            node_vals[2*i] = D.p.params[i][BN[i] + 1]
            node_vals[2*i + 1] = D.p.params[i][BN[i] + 2]

            # if parallel == 31:
            # printf("\nEnd Block 1: %d", <int>i)

        elif vec[i] > node_vals[2*i + 1] and BN[i] != D.p.N[i] - 4: # I believe this has to do with the cubic interpolation points, so this remains 4
            # if parallel == 31:
            # printf("\nExecute block 2: %d", <int>i)
            update_baseNode[i] = 1
            while vec[i] > D.p.params[i][BN[i] + 2]:
                if BN[i] < D.p.N[i] - 4:
                    BN[i] += 1
                elif vec[i] >= D.p.params[i][D.p.N[i] - 1]:
                    vec[i] = D.p.params[i][D.p.N[i] - 1]
                    break
                elif BN[i] == D.p.N[i] - 4:
                    break

            node_vals[2*i] = D.p.params[i][BN[i] + 1]
            node_vals[2*i + 1] = D.p.params[i][BN[i] + 2]

            # if parallel == 31:
            # printf("\nEnd Block 2: %d", <int>i)

        # if parallel == 31:
        # printf("\nTry block 3: %d", <int>i)

        if V_CACHE[i] != vec[i] or update_baseNode[i] == 1:
            # if parallel == 31:
            # printf("\nExecute block 3: %d", <int>i)
            ii = 4*i
            DIFF[ii] = vec[i] - D.p.params[i][BN[i] + 1]
            DIFF[ii] *= vec[i] - D.p.params[i][BN[i] + 2]
            DIFF[ii] *= vec[i] - D.p.params[i][BN[i] + 3]

            DIFF[ii + 1] = vec[i] - D.p.params[i][BN[i]]
            DIFF[ii + 1] *= vec[i] - D.p.params[i][BN[i] + 2]
            DIFF[ii + 1] *= vec[i] - D.p.params[i][BN[i] + 3]

            DIFF[ii + 2] = vec[i] - D.p.params[i][BN[i]]
            DIFF[ii + 2] *= vec[i] - D.p.params[i][BN[i] + 1]
            DIFF[ii + 2] *= vec[i] - D.p.params[i][BN[i] + 3]

            DIFF[ii + 3] = vec[i] - D.p.params[i][BN[i]]
            DIFF[ii + 3] *= vec[i] - D.p.params[i][BN[i] + 1]
            DIFF[ii + 3] *= vec[i] - D.p.params[i][BN[i] + 2]

            # printf("\nupdating V_CACHE")

            V_CACHE[i] = vec[i]

            # if parallel == 31:
            # printf("\nEnd block 3: %d", <int>i)

        # if parallel == 31:
        #     printf("\nTry block 4: %d", <int>i)

        if update_baseNode[i] == 1:
            # if parallel == 31:
            # printf("\nExecute block 4: %d", <int>i)
            # printf("i=%d, ", <int>i)
            # printf("D.p.params[i][BN[i]]: %.2e\n", D.p.params[i][BN[i]])
            CACHE = 1
            SPACE[ii] = 1.0 / (D.p.params[i][BN[i]] - D.p.params[i][BN[i] + 1])
            SPACE[ii] /= D.p.params[i][BN[i]] - D.p.params[i][BN[i] + 2]
            SPACE[ii] /= D.p.params[i][BN[i]] - D.p.params[i][BN[i] + 3]

            SPACE[ii + 1] = 1.0 / (D.p.params[i][BN[i] + 1] - D.p.params[i][BN[i]])
            SPACE[ii + 1] /= D.p.params[i][BN[i] + 1] - D.p.params[i][BN[i] + 2]
            SPACE[ii + 1] /= D.p.params[i][BN[i] + 1] - D.p.params[i][BN[i] + 3]

            SPACE[ii + 2] = 1.0 / (D.p.params[i][BN[i] + 2] - D.p.params[i][BN[i]])
            SPACE[ii + 2] /= D.p.params[i][BN[i] + 2] - D.p.params[i][BN[i] + 1]
            SPACE[ii + 2] /= D.p.params[i][BN[i] + 2] - D.p.params[i][BN[i] + 3]

            SPACE[ii + 3] = 1.0 / (D.p.params[i][BN[i] + 3] - D.p.params[i][BN[i]])
            SPACE[ii + 3] /= D.p.params[i][BN[i] + 3] - D.p.params[i][BN[i] + 1]
            SPACE[ii + 3] /= D.p.params[i][BN[i] + 3] - D.p.params[i][BN[i] + 2]

            # if parallel == 31:
            # printf("\nEnd block 4: %d", <int>i)

        # printf("\ncomputing DIFFs and SPACEs\n")
        # printf("DIFF[ii]: %.2e, ", DIFF[ii])
        # printf("DIFF[ii+1]: %.2e, ", DIFF[ii+1])
        # printf("DIFF[ii+2]: %.2e, ", DIFF[ii+2])
        # printf("DIFF[ii+3]: %.2e\n", DIFF[ii+3])
        
        # printf("SPACE[ii]: %.2e, ", SPACE[ii])
        # printf("SPACE[ii+1]: %.2e, ", SPACE[ii+1])
        # printf("SPACE[ii+2]: %.2e, ", SPACE[ii+2])
        # printf("SPACE[ii+3]: %.2e\n", SPACE[ii+3])

        i += 1

    # printf("Diagnostics: 2\n")
    # for i in range(D.p.ndims):
    #     printf("i=%d, ", <int>i)
    #     printf("vec[i]: %.2e, ", vec[i])
    #     printf("V_CACHE[i]: %.2e\n, ", V_CACHE[i])
    #     printf("D.p.params[i][BN[i]]: %.2e, ", D.p.params[i][BN[i]])
    #     printf("D.p.params[i][BN[i]+1]: %.2e, ", D.p.params[i][BN[i]+1])
    #     printf("D.p.params[i][BN[i]+2]: %.2e, ", D.p.params[i][BN[i]+2])
    #     printf("D.p.params[i][BN[i]+3]: %.2e\n", D.p.params[i][BN[i]+3])

    cdef size_t j, k, l, m, INDEX, II, JJ, KK, LL
    cdef double *address = NULL

    # (4) Here again, I need to iterate over an additional dimension.
    
    # Combinatorics over nodes of hypercube; weight cgs intensities
    # for i in range(4):
    #     II = i * D.p.BLOCKS[0]
    #     for j in range(4):
    #         JJ = j * D.p.BLOCKS[1]
    #         for k in range(4):
    #             KK = k * D.p.BLOCKS[2]
    #             for l in range(4):
    #                 address = D.p.I + (BN[0] + i) * D.p.S[0]
    #                 address += (BN[1] + j) * D.p.S[1]
    #                 address += (BN[2] + k) * D.p.S[2]
    #                 address += BN[3] + l

    #                 temp = DIFF[i] * DIFF[4 + j] * DIFF[8 + k] * DIFF[12 + l]
    #                 temp *= SPACE[i] * SPACE[4 + j] * SPACE[8 + k] * SPACE[12 + l]
    #                 INDEX = II + JJ + KK + l
    #                 if CACHE == 1:
    #                     I_CACHE[INDEX] = address[0]
    #                 I += temp * I_CACHE[INDEX]
    
    # printf("diagnostics for interpolation\n")
    # printf("D.p.S[0]: %d, ", <int>D.p.S[0])
    # printf("BN[0]: %d\n", <int>BN[0])
    # printf("D.p.S[1]: %d, ", <int>D.p.S[1])
    # printf("BN[1]: %d\n", <int>BN[1])
    # printf("D.p.S[2]: %d, ", <int>D.p.S[2])
    # printf("BN[2]: %d\n", <int>BN[2])
    # printf("D.p.S[3]: %d, ", <int>D.p.S[3])
    # printf("BN[3]: %d\n", <int>BN[3])
    # printf("BN[4]: %d\n", <int>BN[4])
        
    
    for i in range(4):
        II = i * D.p.BLOCKS[0]
        for j in range(4):
            JJ = j * D.p.BLOCKS[1]
            for k in range(4):
                KK = k * D.p.BLOCKS[2]
                for l in range(4):
                    LL = l * D.p.BLOCKS[3]
                    for m in range(4):
                        address = D.p.I + (BN[0] + i) * D.p.S[0]
                        address += (BN[1] + j) * D.p.S[1]
                        address += (BN[2] + k) * D.p.S[2]
                        address += (BN[3] + l) * D.p.S[3]
                        address += BN[4] + m
    
                        temp = DIFF[i] * DIFF[4 + j] * DIFF[8 + k] * DIFF[12 + l] * DIFF[16 + m]
                        temp *= SPACE[i] * SPACE[4 + j] * SPACE[8 + k] * SPACE[12 + l] * SPACE[16 + m]
                        INDEX = II + JJ + KK + LL + m
                        if CACHE == 1:
                            I_CACHE[INDEX] = address[0]

                        I += temp * I_CACHE[INDEX]
                        
                        # printf('i=%d,j=%d,k=%d,l=%d,m=%d, ', <int>i, <int>j, <int>k, <int>l, <int>m)
                        # printf('address = %d, ', <int>(address-D.p.I))   
                        # printf('I_CACHE[INDEX] = %d, ', <int>I_CACHE[INDEX])
                        # printf('temp = %0.2e, ', temp)                         
                        # printf('dI = %0.2e\n', temp * I_CACHE[INDEX])

    #if gsl_isnan(I) == 1:
        #printf("\nIntensity: NaN; Index [%d,%d,%d,%d] ",
                #<int>BN[0], <int>BN[1], <int>BN[2], <int>BN[3])

    #printf("\nBase-nodes [%d,%d,%d,%d] ",
                #<int>BN[0], <int>BN[1], <int>BN[2], <int>BN[3])

    if I < 0.0:
        return 0.0
    # Vec here should be the temperature!
    # printf(" I_out: %.8e, ", I * pow(10.0, 3.0 * vec[1]))
    return I * pow(10.0, 3.0 * Temperature)


cdef double eval_hot_norm() nogil:
    # Source radiation field normalisation which is independent of the
    # parameters of the parametrised model -- i.e. cell properties, energy,
    # and angle.
    # Writing the normalisation here reduces the number of operations required
    # during integration.
    # The units of the specific intensity need to be J/cm^2/s/keV/steradian.

    return erg / 4.135667662e-18

