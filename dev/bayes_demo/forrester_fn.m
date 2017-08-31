function [ f ] = forrester_fn( )
%FORRESTER_FN 
f = @(x) ((x.*6-2).^2).*sin((x.*6-2).*2);
end

