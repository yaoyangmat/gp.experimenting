function [y] = joint_sampling_demo(x,hyp)
    n_samples = size(x,1);
    n_dim = size(x,2);
    
    K = kernel(hyp.cov,x);

    z = randn(n_samples, n_dim);
    m = zeros(n_samples, n_dim);
    y = chol(K)'*z + m;
end