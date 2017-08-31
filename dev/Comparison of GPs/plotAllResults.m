% load('all_results.mat')
%load('all_results_20.mat')

figure; hold on;

scatter([fgp_history.time], [fgp_history.rmse], linspace(20,90,numel(fgp_history)), 'd', 'filled', 'MarkerEdgeColor', 'k');
scatter([sgp_history.time], [sgp_history.rmse], linspace(20,90,numel(sgp_history)), 'd', 'filled', 'MarkerEdgeColor', 'k');
scatter([dgp_history.time], [dgp_history.rmse], linspace(20,90,numel(dgp_history)), 'd', 'filled', 'MarkerEdgeColor', 'k');

% plot([fgp_history.time], [fgp_history.rmse],'d','MarkerFaceColor','b');
% plot([sgp_history.time], [sgp_history.rmse],'d','MarkerFaceColor','r');
% plot([dgp_history.time], [dgp_history.rmse],'d','MarkerFaceColor','y');

ylabel('RMSE')
xlabel('Time taken')

[lg, ~] =  legend({'Full', 'Sparse', 'Distributed'},'FontSize',14);
%title('UF1-15: GP validation results')
title('kin-40k: GP validation results')