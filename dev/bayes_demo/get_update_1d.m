function [ acq,x_new ] = get_update_1d( acq_type, gpdata, x_test, f_min, c_gpdata, c_lim, c_type )
%GET_UPDATE_1D 

% In the 1-dimension case, we compute the acquisition function over x_test
% The acquisition function is the improvement criteria * P(feasibility)
% The next update point, x_new is a scalar

acq = zeros(1,size(x_test,1));
for i = 1:length(acq)
    feas = 1;
    for j = 1:length(c_lim)
       feas = feas * compute_feasibility( c_gpdata(j), x_test(i,:), c_lim(j), c_type(j) );
    end
    imp = compute_improvement( acq_type, gpdata, x_test(i,:), f_min );
    acq(i) = imp * feas;
end
[~,idx] = max(acq);
x_new = x_test(idx);

end

