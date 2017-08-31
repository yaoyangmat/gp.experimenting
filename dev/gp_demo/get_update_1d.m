function [ improvement,x_new ] = get_update_1d( acq_type, gpdata, x_test, optimal_y )
%GET_UPDATE 

% In the 1-dimension case, we compute the improvement criteria over x_test
% The next update point, x_new is a scalar

improvement = zeros(1,size(x_test,1));
for i = 1:length(improvement)
    improvement(i) = compute_improvement( acq_type, gpdata, x_test(i,:), optimal_y );
end
[~,idx] = max(improvement);
x_new = x_test(idx);

end

