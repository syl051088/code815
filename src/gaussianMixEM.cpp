#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List gaussianMixEM(const arma::vec &X, int k, int max_iter = 1000, double tol = 1e-6) {
  int n = X.n_elem;
  arma::vec pi_vec(k, arma::fill::ones);
  pi_vec /= k;
  arma::vec mu_vec(k);
  arma::vec xs = arma::sort(X);
  
  for (int j = 0; j < k; ++j) {
    int idx = (j + 1) * n / (k + 1);
    if (idx >= n) { 
      idx = n - 1;
    }
    mu_vec(j) = xs(idx);
  }
  
  double overall_var = arma::var(X, 1); // population variance (normalizing by n)
  arma::vec sigma2_vec(k, arma::fill::value(overall_var));
  
  arma::mat r(n, k, arma::fill::zeros);
  
  double loglik_old = -std::numeric_limits<double>::infinity();
  double loglik_new = 0.0;
  
  // EM iterations
  for (int iter = 0; iter < max_iter; ++iter) {
    loglik_new = 0.0;
    
    // E-step
    for (int i = 0; i < n; ++i) {
      double sum_r = 0.0;
      for (int j = 0; j < k; ++j) {
        double density = 1.0 / std::sqrt(2 * M_PI * sigma2_vec(j)) *
          std::exp(- (X(i) - mu_vec(j)) * (X(i) - mu_vec(j)) / (2 * sigma2_vec(j)));
        double val = pi_vec(j) * density;
        r(i, j) = val;
        sum_r += val;
      }
      
      // Avoid division by zero
      if (sum_r == 0)
        sum_r = 1e-10;
      
      // Normalize responsibilities
      for (int j = 0; j < k; ++j) {
        r(i, j) /= sum_r;
      }
      
      // Accumulate log-likelihood
      loglik_new += std::log(sum_r);
    }
    
    // Check for convergence
    if (std::fabs(loglik_new - loglik_old) < tol) {
      break;
    }
    loglik_old = loglik_new;
    
    // M-step
    for (int j = 0; j < k; ++j) {
      double sum_rj = arma::sum(r.col(j));
      
      // Update mean
      double weighted_sum = arma::dot(r.col(j), X);
      mu_vec(j) = weighted_sum / sum_rj;
      
      // Update variance
      double weighted_var = 0.0;
      for (int i = 0; i < n; ++i) {
        double diff = X(i) - mu_vec(j);
        weighted_var += r(i, j) * diff * diff;
      }
      sigma2_vec(j) = weighted_var / sum_rj;
      
      // Update mixing proportion
      pi_vec(j) = sum_rj / n;
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("pi") = pi_vec,
    Rcpp::Named("mu") = mu_vec,
    Rcpp::Named("sigma2") = sigma2_vec
  );
}
