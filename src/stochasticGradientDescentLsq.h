#include <RcppArmadillo.h>
using namespace std;
using namespace Rcpp;

Rcpp::List stochastic_gradient_descent_lsq(const arma::vec &y,
                                           const arma::mat &A,
                                           const arma::vec &x0,
                                           double lambda,
                                           int batch,
                                           double initial_step_size,
                                           double tol,
                                           int max_iter,
                                           bool printing);