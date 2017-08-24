function [ A,B ] = kernel( hyp,x,z )
%KERNEL uses a squared exponential covariance function with noise and ARD
% hyp = log([ell; sf; sy])
% x.shape = (m,dim)
% z.shape = (n,dim) or z = index of hyperparameter

% USAGE:
% [ K,~ ] = kernel( hyp,x )             % compute covariance matrix
% [ Kss,Kstar ] = kernel( hyp,x,z )     % compute test covariance
% [ dnlz,~ ] = kernel( hyp,x,z )        % compute derivative matrix

dim = size(x,2);
assert(length(hyp) == dim + 2, 'The number of hyperparameters do not match.');

% Extract kernel hyperparameters
ell = exp(hyp(1:dim));
sf = exp(hyp(dim + 1));
sy = exp(hyp(dim + 2));

switch nargin
    case 2                                          % compute covariance matrix = K
        a = diag(1./ell) * x';
        A = sf^2 .* exp(-sqrdist(a,a)/2);  
        A = A + sy^2 .* eye(size(A,1));
        
    case 3
        if nargout == 2                             % compute test covariance = [Kss, Kstar]
            assert(size(x,2) == size(z,2), 'Dimensionality of x & z are not the same.');
            A = sf^2 .* ones(size(z,1),1) + sy^2;   % Kss
            
            a = diag(1./ell) * x';
            b = diag(1./ell) * z';
            B = sf^2 .* exp(-sqrdist(a,b)/2);       % Kstar
        
        else                                        % compute derivative matrix
            if z <= dim                             % lengthscale parameters
               K = kernel(hyp,x);
               a = x(:,z)'/ell(z);
               A = K.*sq_dist(a,a);  
               clear K;
               
            elseif z == dim + 1                     % kernel noise/prior
               K = kernel(hyp,x);
               A = 2*K;
               clear K;
               
            elseif z == dim + 2                     % signal noise
                A = 2*sy^2*eye(size(x,1));
            end
            
        end

end


end

function [ dist ] = sqrdist( a,b )
[D,m] = size(a);
[~,n] = size(b);
dist = zeros(m,n);       % dist.shape = K.shape
for i = 1:D              % accumulate sqr dist by dimensions
    a1 = a(i,:);
    b1 = b(i,:);
    dist = dist + (repmat(a1',1,n) - repmat(b1,m,1)).^2;
end
end