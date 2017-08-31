function [ v_count ] = count_violations(c, c_lim, c_type)
%COUNT_VIOLATIONS

v_matrix = zeros(size(c));
for i = 1:length(c_lim)
    if strcmp(c_type(i), '<')
        v_matrix(:,i) = c(:,i) > c_lim(i);
    elseif strcmp(c_type(i), '>')
        v_matrix(:,i) = c(:,i) < c_lim(i);
    else
        error('Constraint type not recognised!')
    end
end
v_count = sum(v_matrix,2);
