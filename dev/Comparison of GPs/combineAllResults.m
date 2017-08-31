a = load('fgp_sgp_results_20.mat');
b = load('sgp_results_20.mat');

n_a = length([a.fgp_history.rmse]);
n_b = length([b.fgp_history.rmse]);

fields = fieldnames(a);
out = struct;
for i=1:length(fields)
    field = fields{i};
    out.(field).rmse = cat(2, [a.(field).rmse], [b.(field).rmse]);
    out.(field).time = cat(2, [a.(field).time], [b.(field).time]);
    out.(field).params = cat(2, [a.(field).params], [b.(field).params]);
end

save('combined_results_20.mat', 'out')
