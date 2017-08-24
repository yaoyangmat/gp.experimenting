clear all;

% Initialise basic parameters 
n_samples = 100;
n_dim = 20;
ell = rand(n_dim,1); 
sf = 1;
sy = exp(-4);
x = rand(n_samples, n_dim);

% Using GPML covariance function
covfunc = {@covSum, {@covSEard,@covNoise}};
hyp.cov = log([ell; sf; sy]);
hyp.lik = log(sy);
K = feval(covfunc{:}, hyp.cov, x);

% Testing homemade covariance function
K_test = kernel(hyp.cov,x);

% Compare differences
diff = abs(K - K_test);
total_diff = sum(sum(diff));
disp('Total discrepancy = ');
disp(total_diff);
