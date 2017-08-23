function [ ymu, ys2 ] = gp_predict( hyp, cov, x, y, xs )
%GP_PREDICT 
    
    % Hyperparameters
    % hyp.cov = log([ell, sn])
    % hyp.lik = log(sy)
    
    sy2 = exp(hyp.lik*2);
    
    K = cov(hyp.cov,x,x);
    K_star = cov(hyp.cov,x,xs);
    K_ = cov(hyp.cov,xs,xs);
    
    L = chol(K + sy2*eye(length(x)))';
    
    Lk = L\K_star;
    ymu = Lk' * (L\y);
    ys2 = diag(K_) - sum(Lk.^2, 1)';

end

