% load('all_results.mat')
load('all_results_15.mat')

% fgp_history = out.fgp_history;
% sgp_history = out.sgp_history;
% dgp_history = out.dgp_history;

figure;

sz = linspace(30,100,8);
scatter([fgp_history.time], [fgp_history.rmse], sz, 'd', 'filled', 'MarkerEdgeColor', 'k'); hold on;
scatter([sgp_history.time], [sgp_history.rmse], sz, 'd', 'filled', 'MarkerEdgeColor', 'k');
scatter([dgp_history.time], [dgp_history.rmse], sz, 'd', 'filled', 'MarkerEdgeColor', 'k');

% plot([fgp_history.time], [fgp_history.rmse],'d','MarkerFaceColor','b');
% plot([sgp_history.time], [sgp_history.rmse],'d','MarkerFaceColor','r');
% plot([dgp_history.time], [dgp_history.rmse],'d','MarkerFaceColor','y');

ylabel('RMSE')
xlabel('Time taken for optimisation of hyperparameters')

[lg, ~] =  legend('Full', 'Sparse', 'Distributed');
lg.FontSize = 18;

title('UF1-20: GP validation results')