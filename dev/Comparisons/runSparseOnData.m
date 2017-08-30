function [ rmse, time ] = runSparseOnData( MAX_NUM_EVAL, X_train, y_train, X_test, y_test, n_train, n_sparse )
%% Setting up data - training and test
random_order = randperm(length(X_train));
X_train_rand = X_train(random_order,:);
y_train_rand = y_train(random_order,:);
X_train = X_train_rand(1:n_train, :);
y_train = y_train_rand(1:n_train, :);

sparse_random_order = randperm(n_train);
xu_rand = X_train(sparse_random_order,:);
xu = xu_rand(1:n_sparse, :);

%% Setting up cov, mean, inf functions
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
% inf  = @(varargin) inf(varargin{:},struct('s',0.2)); 

% incorporate inducing points and use sparse method
cov = {'apxSparse',cov,xu};
hyp.xu = xu;

%% Sparse GP
% Optimise hyperparameters
fprintf('Optimising hyperparameters for sparse GP...\n')
tic;
hyp = minimize_minfunc(hyp,@gp,-MAX_NUM_EVAL,inf,emptymean,cov,lik,X_train,y_train);
time = toc;

% Generate predictions
[ymu_spgp,ys2_spgp] = gp(hyp,inf,emptymean,cov,lik,X_train,y_train,X_test);
rmse = compute_RMSE(ymu_spgp,y_test);
end