function [varargout] = gp_distributed_dev(hyp, inf, mean, cov, lik, x, y, xs, method)
% x is a struct with M number of arrays to be distributed
M = length(x);

if nargin == 7 % training/optimising hyperparameters
    total_nlZ = zeros(M, 1);
    total_dnlZ = struct;
    total_dnlZ_cov = zeros(M, numel(hyp.cov));
    if ~isfield(hyp,'mean')
        hyp.mean = [];
        total_dnlZ_mean = [];
    else
        total_dnlZ_mean = zeros(M, numel(hyp.mean));
    end
    total_dnlZ_lik = zeros(M, numel(hyp.lik));
    parfor i=1:M
        [nlZ, dnlZ] = gp(hyp, inf, mean, cov, lik, x(i).data, y(i).data);
        total_nlZ(i) = nlZ;
        total_dnlZ_cov(i,:) = dnlZ.cov;
        if ~isempty(hyp.mean)
            total_dnlZ_mean(i,:) = dnlZ.mean;
        end
        total_dnlZ_lik(i,:) = dnlZ.lik;
    end
    total_dnlZ.cov = sum(total_dnlZ_cov, 1)';
    total_dnlZ.mean = total_dnlZ_mean;
    total_dnlZ.lik = sum(total_dnlZ_lik, 1)';
    total_nlZ = sum(total_nlZ, 1);
    varargout = {total_nlZ, total_dnlZ};
elseif nargin >= 8 % inference
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
            s_star2_gpoe = 1./s_star2_inv;
            mu_gpoe = s_star2_gpoe .* sum(beta*(s2.^-1).*mu,2);
            varargout = {mu_gpoe, s_star2_gpoe};
        case 'BCM'
            % BCM implementation
            sy2 = exp(2 * hyp.cov(end-1));      % prior variance
            tmp = (1-M)*sy2.^-1;
            s_star2_inv = sum((s2.^-1),2)+ tmp;
            s_star2_bcm = 1./s_star2_inv;
            mu_bcm = s_star2_bcm .* sum((s2.^-1).*mu,2);
            varargout = {mu_bcm, s_star2_bcm};
        case 'rBCM'
            % rBCM implementation
            sy2 = exp(2 * hyp.cov(end-1));      % prior variance
            beta = 0.5 * (log(sy2) - log(s2));
            tmp = (1 - sum(beta,2))*sy2.^-1;
            s_star2_inv = sum(beta.*(s2.^-1),2)+ tmp;
            s_star2_rbcm = 1./s_star2_inv;
            mu_rbcm = s_star2_rbcm .* sum(beta.*(s2.^-1).*mu,2);
            varargout = {mu_rbcm, s_star2_rbcm};
    end
end
end