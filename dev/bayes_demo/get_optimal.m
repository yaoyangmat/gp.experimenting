function [ x_min, y_min, c_num ] = get_optimal( x, y, c, c_lim, c_type )
%GET_OPTIMAL assumes the optimisation direction is minimization

    v_count = count_violations(c, c_lim, c_type);
    y_min = Inf;
    idx = 1;
    for i = 1:length(y)
        if v_count(i) == 0 && y(i) < y_min
            y_min = y(i);
            idx = i;
        end
    end
    x_min = x(idx, :);
    c_num = v_count(idx);
    
end

