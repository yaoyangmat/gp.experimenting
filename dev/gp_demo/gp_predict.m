function [ ymu, ys2 ] = gp_predict( hyp, cov, x, y, xs )
%GP_PREDICT 
    
    % Hyperparameters
    % hyp.cov = log([ell; sn; sy])
    
    K = cov(hyp,x);
    [Kss, Kstar] = cov(hyp,x,xs);
    
    L = chol(K)';
    
    Lk = L\Kstar;
    ymu = Lk' * (L\y);
    ys2 = Kss - sum(Lk.^2, 1)';

end

