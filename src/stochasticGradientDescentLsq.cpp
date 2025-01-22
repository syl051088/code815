#include "losses.h"
#include "stochasticGradientDescentLsq.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List stochastic_gradient_descent_lsq(const arma::vec &y,
                                           const arma::mat &A,
                                           const arma::vec &x0,
                                           double lambda,
                                           int batch,
                                           double initial_step_size,
                                           double tol,
                                           int max_iter,
                                           bool printing) {
  // Basic checks
  int n = A.n_rows; 
  int p = A.n_cols;
  if ((int)y.n_elem != n) {
    Rcpp::stop("Dimension mismatch: length(y) != nrow(A).");
  }
  if (n < p) {
    Rcpp::stop("Number of observations must be >= number of predictors.");
  }
  
  // Initial subset ("mini-batch") of data
  // randperm(n, k) picks k distinct indices in [0, n-1]
  arma::uvec keep = arma::randperm(n, batch);
  
  arma::mat Asub = A.rows(keep);    // (batch x p)
  arma::vec ysub = y.elem(keep);    // (batch x 1)
  
  // Precompute A'A and A'y for the mini-batch
  arma::mat AA   = Asub.t() * Asub; // (p x p)
  arma::vec Ay   = Asub.t() * ysub; // (p x 1)
  
  // Gradient wrt LS part: (AA*x0 - Ay)/batch
  arma::vec grad = (AA * x0 - Ay) / batch;
  
  // Compute the global loss (over full data) using ridge, then add ridge gradient
  double loss    = loss_ridge(y, A, x0, lambda);
  grad          += 2.0 * lambda * x0 / static_cast<double>(n);
  
  arma::vec x = x0;
  double prev_loss = loss;
  double diff      = std::numeric_limits<double>::infinity();
  
  // For tracking progress
  std::vector<double> diff_rec;
  std::vector<double> loss_rec;
  diff_rec.reserve(max_iter);
  loss_rec.reserve(max_iter);
  
  int iter = 1;
  // Main loop
  while ((iter < max_iter) && (diff > tol)) {
    // Stochastic gradient descent step: step size = initial_step_size / iter
    x = x - (initial_step_size / static_cast<double>(iter)) * grad;
    
    // Re-sample a new mini-batch
    keep  = arma::randperm(n, batch);
    Asub  = A.rows(keep);
    ysub  = y.elem(keep);
    
    // Recompute the gradient on this new batch
    AA    = Asub.t() * Asub;
    Ay    = Asub.t() * ysub;
    grad  = (AA * x - Ay) / batch;
    
    // Compute loss on the FULL dataset
    loss  = loss_ridge(y, A, x, lambda);
    
    // Add the ridge portion of the gradient
    grad += 2.0 * lambda * x / static_cast<double>(n);
    
    // Relative improvement
    double rel   = (prev_loss - loss) / std::fabs(prev_loss);
    diff         = std::fabs(rel);
    
    // Record for output
    diff_rec.push_back(rel);
    loss_rec.push_back(loss);
    
    prev_loss = loss;
    iter++;
  }
  
  if (printing) {
    Rcpp::Rcout << "Converged after " << iter << " steps (or reached max_iter).\n";
  }
  
  // Return results in a list
  return Rcpp::List::create(
    Rcpp::Named("x")    = x,                 // final coefficient vector
    Rcpp::Named("diff") = Rcpp::wrap(diff_rec),
    Rcpp::Named("loss") = Rcpp::wrap(loss_rec)
  );
}