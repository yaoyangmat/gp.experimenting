function [ x_min, f_min, c_num ] = get_optimal( x, y, c, c_lim, c_type )
%GET_OPTIMAL assumes the optimisation direction is minimization
    
    if isempty(c_lim)
        [f_min, idx] = min(y);
        x_min = x(idx,:);
        c_num = 0;
        
    else
        v_count = count_violations(c, c_lim, c_type);
        f_min = Inf;
        idx = 1;
        for i = 1:length(y)
            if v_count(i) == 0 && y(i) < f_min
                f_min = y(i);
                idx = i;
            end
        end
        x_min = x(idx, :);
        c_num = v_count(idx);
    end
    
end

