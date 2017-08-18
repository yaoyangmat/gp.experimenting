Methods = {'Old';'Full';'Sparse'};
RMSE = [mean(diff_old); mean(diff_full); mean(diff_sparse)];
Time = [mean(hyperparam_old_time); mean(hyperparam_full_time); mean(hyperparam_sparse_time)];
T = table(RMSE,Time,'RowNames',Methods);
disp(T)
