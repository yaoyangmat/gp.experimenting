function [ improvement ] = get_improvement_criteria( Action, gpdata, X, Y_MIN )
%GET_IMPROVEMENT_CRITERIA 
    [ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, X);
    ys = sqrt(ys2);
    improvement = zeros(size(X,1),1);
    pd = makedist('Normal',0,1);        % Standard normal
    
    for i = 1:length(improvement)
        x = (Y_MIN-ymu(i))/ys(i);
        switch Action
            case 'P_improvement'
                improvement(i) = cdf(pd,x);
            case 'E_improvement'
                improvement(i) = (Y_MIN-ymu(i))*cdf(pd,x) + ys(i)*pdf(pd,x) ;
        end
    end

end

