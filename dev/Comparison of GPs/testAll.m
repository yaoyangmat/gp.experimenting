function testAll()
%% Shared params
n_test = 5000;
n_dim = 20;
n_responses = 2;

%% Params for Full GP
full_MAX_NUM_EVAL = 50;
full_n_train = [3000; 3500; 4000; 4500; 5000; 5500];

%% Params for Sparse GP
sparse_MAX_NUM_EVAL = 200;
sparse_n_train = [9000; 10000];
n_sparse = sparse_n_train/10;

%% Params for Distributed GP
dist_MAX_NUM_EVAL = 150;
dist_n_train = [8000; 10000];
M = [4; 4];

%% RUN AND SAVE
% fgp_history = run(full_MAX_NUM_EVAL, full_n_train, n_test, n_dim, n_responses, 'runFullUF1');
% sgp_history = run(sparse_MAX_NUM_EVAL, sparse_n_train, n_test, n_dim, n_responses, 'runSparseUF1', n_sparse);
dgp_history = run(dist_MAX_NUM_EVAL, dist_n_train, n_test, n_dim, n_responses, 'runDistUF1', M);

save('dgp_results_20.mat', 'dgp_history')

end

%% Helper
function history = run(MAX_NUM_EVAL, n_train, n_test, n_dim, n_responses, func_name, m)
    for i=1:length(n_train)
        if strcmp(func_name, 'runFullUF1')
            [ rmse, time ] = feval( func_name, MAX_NUM_EVAL, n_train(i), n_test, n_dim, n_responses );
        else
            [ rmse, time ] = feval( func_name, MAX_NUM_EVAL, n_train(i), n_test, n_dim, n_responses, m(i) );
        end
        history(i).rmse = rmse;
        history(i).time = time;
        history(i).params.num_eval = MAX_NUM_EVAL;
        history(i).params.n_train = n_train(i);
        if strcmp(func_name, 'runSparseUF1')
            history(i).params.n_sparse = m(i);
        elseif strcmp(func_name, 'runDistUF1')
            history(i).params.M = m(i);
        end
    end
end