function [ improvement ] = compute_improvement( acq_type, gpdata, X, F_MIN )
%COMPUTE_IMPROVEMENT
    [ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, X);
    ys = sqrt(ys2);
    pd = makedist('Normal',0,1);        % Standard normal

    x = (F_MIN-ymu)/ys;
    switch acq_type
        case 'PI'
            improvement = cdf(pd,x);
        case 'EI'
            improvement = (F_MIN-ymu)*cdf(pd,x) + ys*pdf(pd,x) ;
    end

end

