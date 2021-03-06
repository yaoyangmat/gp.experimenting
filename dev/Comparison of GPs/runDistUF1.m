function [ rmse, time ] = runDist( MAX_NUM_EVAL, n_train, n_test, n_dim, n_responses, M )
%% Setting up data - training and test
% Create training data
lb = zeros(1,n_dim);
lb(2:end) = lb(2:end) - 1;
ub = ones(1,n_dim);
X_train = lhs(lb,ub,n_train);

tmp_y = zeros(n_train,n_responses);
for i=1:n_train
    tmp_y(i,:) = UF1(X_train(i,:)')';
end
y_train = tmp_y(:,1);   % Only test on the first response, z_1

% Randomly split training data into M sets for distributed GP. ASSUMPTION: X IS >1 DIMENSIONAL
random_order = randperm(n_train);
X_train_rand = X_train(random_order,:);
y_train_rand = y_train(random_order,:);
X_train_DGP = split_data(X_train_rand, M);
y_train_DGP = split_data(y_train_rand, M);

% Create test data
X_test = [rand(n_test,1), rand(n_test,n_dim-1)*2-1];
tmp_ys = zeros(n_test,2);
for i=1:n_test
    tmp_ys(i,:) = UF1(X_test(i,:)')';
end
y_test = tmp_ys(:,1);

%% Initialize hyp, cov, mean, lik
% Initialise guess: logtheta0 (from Swordfish)
stdX = std(X_train)';
stdX( stdX./abs(mean(X_train))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(y_train); 0.05*std(y_train)]);

% Use covariance function (from Swordfish)
hyp=[];inf=[];lik=[];cov=[];
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;  
emptymean = [];
lik = {@likGauss};    
hyp.lik = logtheta0(end);
inf = @infGaussLik;

%% Optimise hyperparameters for distributed GP
fprintf('Optimising hyperparameters for distributed GP...\n')
tic;
hyp = minimize_minfunc(hyp,@gp_distributed_dev,-MAX_NUM_EVAL,inf,emptymean,cov,lik,X_train_DGP,y_train_DGP);
time = toc;

[ymu_dgp,ys2_dgp] = gp_distributed_dev(hyp,inf,emptymean,cov,lik,X_train_DGP,y_train_DGP,X_test,'rBCM');
rmse = compute_RMSE(ymu_dgp,y_test);

end

%% HELPER
function [X_split] = split_data(X, M)
    len = length(X);
    X_split = struct;
    set_size = ceil(len/M);
    for i=1:M
        start_index = (i-1)*set_size + 1;
        end_index = i*set_size;
        end_index = min(end_index, len);
        X_split(i).data = X(start_index:end_index, :);
    end
end

