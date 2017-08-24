function [ nlZ, dnlZ ] = gp_train( hyp, cov, x, y )
%GP_TRAIN
    % Hyperparameters
    % hyp.cov = log([ell, sn])
    % hyp.lik = log(sy)
    
    n = size(x,1);
    K = cov(hyp.cov,x);
    
    L = chol(K)';
    alpha = L'\(L\y);
    
    nlZ = 0.5*y'*alpha + 0.5*log(diag(L)) + 0.5*n*log(2*pi);
    
    dnlZ = zeros(size(hyp.cov));        % compute partial derivatives
    W = L'\(L\eye(n))-alpha*alpha';  
    for i = 1:length(hyp.cov)
        a = cov(hyp.lik, x, i);
        dnlZ(i) = sum(sum(W.*a))/2;
    end
end

