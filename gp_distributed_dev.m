function [varargout] = gp_distributed_dev(hyp, inf, mean, cov, lik, x, y, xs, method)
% x is a struct with M number of arrays to be distributed
M = length(x);

if nargin == 7 % training/optimising hyperparameters
    for i=1:M
        [nlZ, dnlZ] = gp(hyp, inf, mean, cov, lik, x(i).data, y(i).data);
        if i == 1
            total_nlZ = nlZ;
            total_dnlZ = dnlZ;
        else
            total_nlZ = total_nlZ + nlZ;
            total_dnlZ.cov = total_dnlZ.cov + dnlZ.cov;  % this is quite hardcoded
            total_dnlZ.mean = total_dnlZ.mean + dnlZ.mean;
            total_dnlZ.lik = total_dnlZ.lik + dnlZ.lik;
        end
    end
    varargout = {total_nlZ, total_dnlZ};
elseif nargin >= 8
    n_test = length(xs);
    mu = zeros(n_test,M);
    s2 = zeros(n_test,M);
    for i=1:M
        [ymu, ys2, ~, ~] = gp(hyp, inf, mean, cov, lik, x(i).data, y(i).data, xs);
        mu(:,i) = ymu;
        s2(:,i) = ys2;
    end
    if nargin == 8
        method = 'rBCM';
    end
    switch (method)
        case 'PoE'
            % PoE implementation
            s_star2_inv = sum((s2.^-1),2);
            s_star2_poe = 1./s_star2_inv;
            mu_poe = s_star2_poe .* sum((s2.^-1).*mu,2);
            varargout = {mu_poe, s_star2_poe};
        case 'gPoE'
            % gPoE implementation
            beta = 1/M;
            s_star2_inv = sum(beta*(s2.^-1),2);
            s_star2_poe = 1./s_star2_inv;
            mu_poe = s_star2_poe .* sum(beta*(s2.^-1).*mu,2);
            varargout = {mu_poe, s_star2_poe};
        case 'BCM'
            % BCM implementation
            sy2 = exp(2 * hyp.cov(end-1));      % prior variance
            tmp = (1-M)*sy2.^-1;
            s_star2_inv = sum((s2.^-1),2)+ tmp;
            s_star2_poe = 1./s_star2_inv;
            mu_poe = s_star2_poe .* sum((s2.^-1).*mu,2);
            varargout = {mu_poe, s_star2_poe};
        case 'rBCM'
            sy2 = exp(2 * hyp.cov(end-1));      % prior variance
            beta = 0.5 * (log(sy2) - log(s2));
            tmp = (1 - sum(beta,2))*sy2.^-1;
            s_star2_inv = sum(beta.*(s2.^-1),2)+ tmp;
            s_star2_poe = 1./s_star2_inv;
            mu_poe = s_star2_poe .* sum(beta.*(s2.^-1).*mu,2);
            varargout = {mu_poe, s_star2_poe};
    end
end
end