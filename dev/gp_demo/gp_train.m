function [ nlZ, dnlZ ] = gp_train( hyp, cov, x, y )
%GP_TRAIN
    % Hyperparameters
    % hyp = log([ell; sn; sy])
    
    n = size(x,1);
    K = cov(hyp,x);
    
    L = chol(K)';
    alpha = L'\(L\y);
    
    nlZ = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);
    
    dnlZ = zeros(size(hyp));        % compute partial derivatives
    W = L'\(L\eye(n))-alpha*alpha';  
    for i = 1:length(hyp)
        a = cov(hyp, x, i);
        dnlZ(i) = sum(sum(W.*a))/2;
    end
end

