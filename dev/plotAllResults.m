% load('all_results.mat')
load('all_results_15.mat')

% fgp_history = out.fgp_history;
% sgp_history = out.sgp_history;
% dgp_history = out.dgp_history;

figure;
hold on;
plot([fgp_history.time], [fgp_history.rmse],'d','MarkerFaceColor','b');
plot([sgp_history.time], [sgp_history.rmse],'d','MarkerFaceColor','r');
plot([dgp_history.time], [dgp_history.rmse],'d','MarkerFaceColor','y');
ylabel('RMSE')
%ylim([0 0.02])
xlabel('Time taken for optimisation of hyperparameters')
lg = legend('Full', 'Sparse', 'Distributed');
lg.FontSize = 14;
title('UF1-15: GP validation results')