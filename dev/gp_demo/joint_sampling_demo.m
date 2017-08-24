function [y] = joint_sampling_demo(x,hyp)
    n_samples = size(x,1);
    n_dim = size(x,2);
    
    sy2 = exp(hyp.lik *2);
    
    K = kernel(hyp.cov,x,x);

    z = randn(n_samples, n_dim);
    m = zeros(n_samples, n_dim);
    y = chol(K + sy2*eye(n_samples))'*z + m;
end