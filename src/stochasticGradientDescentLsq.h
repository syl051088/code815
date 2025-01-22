#include <RcppArmadillo.h>
using namespace std;
using namespace Rcpp;

Rcpp::List stochastic_gradient_descent_lsq(const arma::vec &y,
                                           const arma::mat &A,
                                           const arma::vec &x0,
                                           double lambda,
                                           int batch,
                                           double initial_step_size = 1.0,
                                           double tol              = 1e-6,
                                           int max_iter            = 10000,
                                           bool printing           = false);