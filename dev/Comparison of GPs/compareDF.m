function compareDF()
%% Preparing data
data = load('kin40k.mat');
X_train = data.x;
y_train = data.y;
X_test = data.xtest;
y_test = data.ytest;

%% Params for Full GP
full_MAX_NUM_EVAL = 100;
% full_n_train = [500; 1000; 1500; 2000; 2500; 3000; 3500; 4000; 4500];
% full_n_train = [3000; 5000];
% full_n_train = [5000];

%% Params for Sparse GP
% sparse_MAX_NUM_EVAL = 200;
% sparse_n_train = [10000; 10000; 10000; 10000; 10000; 10000; 10000; 10000; 10000; 10000];
% n_sparse = [50; 100; 200; 300; 400; 500; 700; 900; 1200];
% sparse_n_train = [10000; 10000; 10000; 10000];
% n_sparse = [50; 100; 200; 300; ];

%% Params for Distributed GP
dist_MAX_NUM_EVAL = 100;
% dist_n_train = [700000];
% M = [170];
dist_n_train = [10000];
M = [6]; % no. of experts

%% RUN AND SAVE
% fgp_history = runOnData(full_MAX_NUM_EVAL, X_train, y_train, X_test, y_test, full_n_train, 'runFullOnData');
dgp_history = runOnData(dist_MAX_NUM_EVAL, X_train, y_train, X_test, y_test, dist_n_train, 'runDistOnData', M);

save('DFcomparison2.mat', 'dgp_history')

end

%% Helper
function history = runOnData(MAX_NUM_EVAL, X_train, y_train, X_test, y_test, n_train, func_name, m)
    for i=1:length(n_train)
        if strcmp(func_name, 'runFullOnData')
            [ rmse, time ] = feval( func_name, MAX_NUM_EVAL, X_train, y_train, X_test, y_test, n_train(i) );
        else
            [ rmse, time ] = feval( func_name, MAX_NUM_EVAL, X_train, y_train, X_test, y_test, n_train(i), m(i) );
        end
        history(i).rmse = rmse;
        history(i).time = time;
        history(i).params.num_eval = MAX_NUM_EVAL;
        history(i).params.n_train = n_train(i);
        if strcmp(func_name, 'runSparseOnData')
            history(i).params.n_sparse = m(i);
        elseif strcmp(func_name, 'runDistOnData')
            history(i).params.M = m(i);
        end
    end
end