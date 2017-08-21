disp('-------------------------------')
disp('Testing Distributed GP vs Full GP...')
disp('-------------------------------')
close all; clear all;
%% Basic parameters
MAX_NUM_EVAL = 10;         % Maximum allowed function evals
n_train = 20;             % Number of training points
n_test = 500;              % Number of test points
M = 10;                     % Number of sets for distributed GP
%sn = 0.001;                   % Noise standard deviation. NOT INCLUDING NOISE for now (CHECK THIS OUT!!)
%% Setting up data - training and test
X_train = linspace(-5,5,n_train)';
y_train = sin(X_train);

% Randomly split training data into M sets for distributed GP
random_order = randperm(n_train);
X_train_rand = X_train(random_order);
y_train_rand = y_train(random_order);
X_train_DGP = split_1d_data(X_train_rand, M);
y_train_DGP = split_1d_data(y_train_rand, M);

X_test = linspace(-15, 15, n_test)';
y_test = sin(X_test);

%% Initialize hyp, cov, mean, lik
% Initialise guess: logtheta0 (from Swordfish)
stdX = std(X_train)';
stdX( stdX./abs(mean(X_train))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(y_train); 0.05*std(y_train)]);

hyp=[];inf=[];lik=[];cov=[];
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;  
emptymean = [];
lik = {@likGauss};    
hyp.lik = logtheta0(end);
inf = @infGaussLik;

%% Use Full GP
fprintf('Optimising hyperparameters for full GP...\n')
tic;
hyp = minimize_minfunc(hyp,@gp,-MAX_NUM_EVAL,inf,emptymean,cov,lik,X_train,y_train);      
time_full = toc;

% Generate predictions
[ymu,ys2] = gp(hyp,inf,emptymean,cov,lik,X_train,y_train,X_test);                 % dense prediction
rmse_full = compute_RMSE(ymu,y_test);

%% Initialize hyp, cov, mean, lik again
% Initialise guess: logtheta0 (from Swordfish)
stdX = std(X_train)';
stdX( stdX./abs(mean(X_train))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(y_train); 0.05*std(y_train)]);

hyp=[];inf=[];lik=[];cov=[];
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;
emptymean = [];
lik = {@likGauss};    
hyp.lik = logtheta0(end);
inf = @infGaussLik;

%% Optimise hyperparameters for distributed GP
methods = {'PoE', 'gPoE', 'BCM', 'rBCM'};
% ymu_dgp_store = zeros(n_test, length(methods));
% ys2_dgp_store = zeros(n_test, length(methods));
% rmse_dgp_store = zeros(length(methods));
dgp_history = struct;
for method = methods
    fprintf('Optimising hyperparameters...\n')
    tic;
    hyp = minimize_minfunc(hyp,@gp_distributed_dev,-MAX_NUM_EVAL,inf,emptymean,cov,lik,X_train_DGP,y_train_DGP);
    time_dgp = toc;

    [ymu_dgp,ys2_dgp] = gp_distributed_dev(hyp,inf,emptymean,cov,lik,X_train_DGP,y_train_DGP,X_test,method{1});
    rmse_dgp = compute_RMSE(ymu_dgp,y_test);
    
    dgp_history.(method{1}).ymu = ymu_dgp;
    dgp_history.(method{1}).ys2 = ys2_dgp;
    dgp_history.(method{1}).rmse = rmse_dgp;
    dgp_history.(method{1}).time = time_dgp;
end
%% Printing results
% fprintf('Validation results performed on %d test points...\n', n_test)
% fprintf('RMSE for full GP: %f\n', rmse_full)
% fprintf('RMSE for distributed GP: %f\n', rmse_dgp)
% fprintf('Time taken for full GP: %f\n', time_full)
% fprintf('Time taken for distributed GP: %f\n', time_dgp)

%% Plot results
figure;
ys = sqrt(ys2);
for i=1:length(methods)
    subplot(1,4,i);
    hold on;
    ymu_dgp = dgp_history.(methods{i}).ymu;
    ys2_dgp = dgp_history.(methods{i}).ys2;
    ys_dgp = sqrt(ys2_dgp);
    plot(X_test, ymu_dgp);
    plot(X_test, ymu);
    scatter(X_train, y_train, 'x');
    jbfill(X_test', ymu_dgp'+2*ys_dgp', ymu_dgp'-2*ys_dgp', 'b', 'k', 1, 0.2);
    jbfill(X_test', ymu'+2*ys', ymu'-2*ys', 'g', 'k', 1, 0.2);
    title(methods{i});
    legend('Distributed', 'Full');
    hold off;
end
foo = 1;

%% HELPER FUNCTIONS
function [X_split] = split_1d_data(X, M)
    len = length(X);
    X_split = struct;
    set_size = ceil(len/M);
    for i=1:M
        start_index = (i-1)*set_size + 1;
        end_index = i*set_size;
        end_index = min(end_index, len);
        X_split(i).data = X(start_index:end_index);
    end
end