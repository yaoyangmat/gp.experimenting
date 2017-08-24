function [ K ] = kernel( hyp,x,z )
%KERNEL uses a simple squared exponential covariance function
% X.shape = (m,dim)
% Z.shape = (n,dim)
% K.shape = (m,n)

    assert(size(x,2) == size(z,2), 'Dimensionality of x & z are not the same.');
    dim = size(x,2);
    assert(length(hyp) == dim + 1, 'The number of hyperparameters do not match.');

    % Extract kernel hyperparameters
    ell = exp(hyp(1:dim));
    sn = exp(hyp(dim + 1));

    a = diag(1./ell) * x';
    b = diag(1./ell) * z';

    K = sn.^2 .* exp(-sqrdist(a,b)/2);

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