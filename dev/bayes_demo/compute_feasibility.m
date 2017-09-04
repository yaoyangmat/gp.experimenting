function [ feasibility ] = compute_feasibility( gpdata, X, C_LIM, C_TYPE )
%COMPUTE_FEASIBILITY 
    [ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, X);
    ys = sqrt(ys2);
    pd = makedist('Normal',0,1);        % Standard normal

    if strcmp(C_TYPE, '<')
        x = (C_LIM-ymu)/ys;
    elseif strcmp(C_TYPE, '>')
        x = -1*(C_LIM-ymu)/ys;
    else
        error('Constraint type not recognised!')
    end
    feasibility = cdf(pd,x);

end

