clear all, close all
write_fig = 0;

% meanfunc = {@meanSum, {@meanLinear, @meanConst}}; 
% hyp.mean = [0.5; 1];
% 
% covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; 
% hyp.cov = log([ell; sf]);

meanfunc = {[]};
hyp.mean = [];

covfunc = {@covSum, {@covPeriodic, @covNoise}};
ell = 2.4; p = 1; sf = 1; sy = 1e-6;
hyp.cov = log([ell; p; sf; sy]);

likfunc = @likGauss; sn = 1e-4; hyp.lik = log(sn);

n = 50;
lb = -5; ub = 5;
x = linspace(-5,5,n)';

K = feval(covfunc{:}, hyp.cov, x);
%mu = feval(meanfunc{:}, hyp.mean, x);
mu = 0;
y = chol(K)'*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);
 
figure;
set(gca, 'FontSize', 24)
plot(x, y, 'MarkerSize', 12)
xlabel('input, x')
ylabel('output, y')