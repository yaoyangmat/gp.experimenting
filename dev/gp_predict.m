function [ mu, s2 ] = gp_predict( hyp, cov, X, y, X_star )
%GP_PREDICT 
    if isempty(X)
        mu = 0;
        s2 = 1;
        return
    end
    
    s = 1e-6; % stabilising noise
    K = cov(hyp,X,X);
    L = chol(K + s*eye(length(X)));
    
    Lk = L\(cov(hyp,X,X_star));
    mu = Lk' * (L\y);
    
    K_ = cov(hyp,X_star,X_star);
    s2 = diag(K_) - sum(Lk.^2, 1)';
    

end

