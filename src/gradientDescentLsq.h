#include <RcppArmadillo.h>
using namespace std;
using namespace Rcpp;

Rcpp::List gradient_descent_lsq(const arma::vec & y,
                          const arma::mat & A,
                          const arma::vec & x0,
                          double lambda,
                          double gamma,
                          double tol,
                          int max_iter,
                          bool printing);
  