function [response] = get_response(fn_array, x)
%GET_RESPONSE applies the functions within fn_array to x

    if isempty(fn_array)
        response = [];
        return
    end

    n_fn = numel(fn_array);
    response = zeros(size(x,1), n_fn);
    for i = 1:n_fn
        f = fn_array{i};
        response(:,i) = f(x);
    end
end
