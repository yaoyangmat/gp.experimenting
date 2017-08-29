clear all, close all;

%% Basic parameters
MAX_NUM_EVAL_FULL = 50;         % Maximum allowed function evals for full GP
n_train = 5000;                  % Number of training points
n_test = 5000;                  % Number of test points
n_dim = 20;                      % Size of UF1 problem
n_responses = 2 ;               % Number of responses for UF1 problem

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
% k1 = {@covSum, {@covSEard,@covNoise}};
% k2 = {@covPERiso, {@covSEisoU}};
% cov = {@covSum, {k1, k2}}; 
% hyp.cov = [logtheta0 ; 1 ; 1];  
k1 = {@covSum, {@covSEard,@covNoise}};
cov = k1; 
hyp.cov = logtheta0;
emptymean = [];
lik = {@likGauss};    
hyp.lik = logtheta0(end);
inf = @infGaussLik;

%% Full GP
% Optimise hyperparameters
fprintf('Optimising hyperparameters for full GP...\n')
tic;
hyp = minimize_minfunc(hyp,@gp,-MAX_NUM_EVAL_FULL,inf,emptymean,cov,lik,X_train,y_train);      
time_full = toc;

% Generate predictions
[nlZ1,dnlZ1] = gp(hyp,inf,emptymean,cov,lik,X_train,y_train); nlZ1
[ymu,ys2] = gp(hyp,inf,emptymean,cov,lik,X_train,y_train,X_test);                 % dense prediction
rmse_full = compute_RMSE(ymu,y_test);
disp('RMSE error on response');
disp(rmse_full);