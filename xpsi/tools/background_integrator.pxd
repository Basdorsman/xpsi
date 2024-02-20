from GSL cimport (gsl_function,
                   gsl_integration_cquad,
                   gsl_integration_cquad_workspace,
                   gsl_integration_cquad_workspace_alloc,
                   gsl_integration_cquad_workspace_free)

ctypedef gsl_integration_cquad_workspace gsl_cq_work
