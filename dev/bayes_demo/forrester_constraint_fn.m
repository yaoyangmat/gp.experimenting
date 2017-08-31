function [ f ] = forrester_constraint_fn( )
%FORRESTER_CONSTRAINT_FN 
pd1 = makedist('Normal',0.9,0.25);
pd2 = makedist('Normal',0,0.45);
f = @(x) 1.3 - pdf(pd1,x)- pdf(pd2,x);          % Constraint function

end

