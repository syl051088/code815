#include "gradientDescentLsq.h"
#include "losses.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List gradient_descent_lsq(const arma::vec & y,
                          const arma::mat & A,
                          const arma::vec & x0,
                          double lambda,
                          double gamma,
                          double tol,
                          int max_iter,
                          bool printing) {
  
  // Precompute
  arma::mat AA = A.t() * A;     // p x p
  arma::vec Ay = A.t() * y;     // p x 1
  
  // Initialize
  arma::vec x      = x0;
  arma::vec grad   = AA * x0 - Ay;           // gradient for the LSQ part
  double    loss   = loss_ridge(y, A, x0, lambda);
  grad            += 2.0 * lambda * x0;      // add ridge part: 2 * lambda * x0
  
  double prev_loss = loss;
  double diff      = std::numeric_limits<double>::infinity();
  
  // We'll store iteration-wise values in std::vectors for convenience
  std::vector<double> diff_rec;
  std::vector<double> loss_rec;
  diff_rec.reserve(max_iter);
  loss_rec.reserve(max_iter);
  
  // Iteration
  int iter = 1;
  while ((iter < max_iter) && (diff > tol)) {
    // Gradient descent update
    x    = x - gamma * grad;
    
    // Recompute gradient and loss
    grad = AA * x - Ay;
    loss = loss_ridge(y, A, x, lambda);
    grad = grad + 2.0 * lambda * x;
    
    // Relative improvement
    double rel = (prev_loss - loss) / std::fabs(prev_loss);
    diff       = std::fabs(rel);
    
    // Record
    diff_rec.push_back(rel);
    loss_rec.push_back(loss);
    
    // Update
    prev_loss = loss;
    iter++;
  }
  
  if (printing) {
    Rcout << "Converged after " << iter << " steps.\n";
  }
  
  // Return results as a List (similar to the original R function)
  return List::create(
    Named("x")    = x,                // final coefficients
    Named("diff") = wrap(diff_rec),   // vector of relative differences
    Named("loss") = wrap(loss_rec)    // vector of loss function values
  ); 
  
}
