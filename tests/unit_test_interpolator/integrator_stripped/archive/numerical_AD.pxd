from preload cimport _preloaded
#from libcpp.vector cimport vector
from libcpp cimport bool

# cdef double eval_hot(size_t THREAD,
#                      double E,
#                      double mu,
#                      const double *const VEC,
#                      void *const data) nogil

# cdef double eval_hot(size_t THREAD,
#                      double E,
#                      double mu,
#                      double g,
#                      double T,
#                      void *const data) nogil

cdef double eval_hot(size_t THREAD,
                      double E,
                      double mu,
                      double tau,
                      double t_bb,
                      double t_e,
                      void *const data) nogil

cdef double eval_hot_norm() nogil

cdef void* init_hot(size_t numThreads, const _preloaded *const preloaded) nogil

cdef int free_hot(size_t numThreads, void *const data) nogil

ctypedef struct mystruct:
    double* temp2
    double* I_CACHE
    double* I_temp

cdef mystruct eval_hot_seploops(size_t THREAD,
                      double E,
                      double mu,
                      double tau,
                      double t_bb,
                      double t_e,
                      void *const data)

# cdef bint compare_double(double x, double y, double epsilon = 1.0e-12) nogil

# cdef vector[double] multiply_vectors(int length, vector[double] vexy, vector[double] vex, vector[double] vey)

# cdef double* multiply_pointers(double* vecxy, double* vecx, double* vecy, int length)
# cdef double* add_pointers(double* vecxy, double* vecx, double* vecy, int length)

# cdef void simd_add(float[:] a, float[:] b, float[:] c) nogil