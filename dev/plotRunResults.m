load('RMSE log.mat')

data = [diff_old; diff_full; diff_sparse]';
averages = mean(data, 1);
stddevs = std(data, 0, 1);

figure;
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8])
b = bar(data);
l{1} = strcat('Old GP: mu=', num2str(averages(1)), ', sigma=', num2str(stddevs(1))); 
l{2} = strcat('New full GP: mu=', num2str(averages(2)), ', sigma=', num2str(stddevs(2)));
l{3} = strcat('New sparse GP: mu=', num2str(averages(3)), ', sigma=', num2str(stddevs(3)));

legend(b,l);