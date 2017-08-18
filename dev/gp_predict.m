function [ mu, s2 ] = gp_predict( hyp, cov, X, y, X_star )
%GP_PREDICT 
    
    s = 0;%1e-6; % stabilising noise
    K = cov(hyp,X,X);
    K_star = cov(hyp,X,X_star);
    K_ = cov(hyp,X_star,X_star);
    
    L = chol(K + s*eye(length(X)),'lower');
    
    Lk = L\K_star;
    mu = Lk' * (L\y);
    s2 = diag(K_) - sum(Lk.^2, 1)';
    

end

