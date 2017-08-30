function [y] = gp_joint_sampling(x,hyp,cov)
    n_samples = size(x,1);
    n_dim = size(x,2);
    
    K = cov(hyp,x);

    z = randn(n_samples, n_dim);
    m = zeros(n_samples, n_dim);
    y = chol(K)'*z + m;
end