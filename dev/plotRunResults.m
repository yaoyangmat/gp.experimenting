load('RMSE log 10 dim 3000 n.mat')

data = [diff_old; diff_full; diff_sparse]';
averages = mean(data, 1);
stddevs = std(data, 0, 1);

figure(1);
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8])
b = bar(data);
l{1} = strcat('Old GP: \mu=', num2str(averages(1),2), ', \sigma=', num2str(stddevs(1),2)); 
l{2} = strcat('New full GP: \mu=', num2str(averages(2),2), ', \sigma=', num2str(stddevs(2),2));
l{3} = strcat('New sparse GP: \mu=', num2str(averages(3),2), ', \sigma=', num2str(stddevs(3),2));
ylabel('RMSE');
xlabel('Trial number');
legend(b,l,'FontSize',14);
% legend({'One','Two','Three','Four'},'FontSize',14)

% figure(2);
% set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8])
