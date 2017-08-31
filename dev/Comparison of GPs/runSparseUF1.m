function [ rmse, time ] = runSparse( MAX_NUM_EVAL, n_train, n_test, n_dim, n_responses, n_sparse )
%% Setting up data - training and test
% Create training data
lb = zeros(1,n_dim);
lb(2:end) = lb(2:end) - 1;
ub = ones(1,n_dim);
X_train = lhs(lb,ub,n_train);
xu = lhs(lb,ub,n_sparse);

tmp_y = zeros(n_train,n_responses);
for i=1:n_train
    tmp_y(i,:) = UF1(X_train(i,:)')';
end
y_train = tmp_y(:,1);   % Only test on the first response, z_1

% Create test data
X_test = [rand(n_test,1), rand(n_test,n_dim-1)*2-1];
tmp_ys = zeros(n_test,2);
for i=1:n_test
    tmp_ys(i,:) = UF1(X_test(i,:)')';
end
y_test = tmp_ys(:,1);

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

