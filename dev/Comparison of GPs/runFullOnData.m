function [ rmse, time ] = runFullOnData( MAX_NUM_EVAL, X_train, y_train, X_test, y_test, n_train )
%% Setting up data - training and test
random_order = randperm(length(X_train));
X_train_rand = X_train(random_order,:);
y_train_rand = y_train(random_order,:);
X_train = X_train_rand(1:n_train, :);
y_train = y_train_rand(1:n_train, :);

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

%% Full GP
% Optimise hyperparameters
fprintf('Optimising hyperparameters for full GP...\n')
tic;
hyp = minimize_minfunc(hyp,@gp,-MAX_NUM_EVAL,inf,emptymean,cov,lik,X_train,y_train);      
time = toc;

% Generate predictions
[ymu,ys2] = gp(hyp,inf,emptymean,cov,lik,X_train,y_train,X_test);                 % dense prediction
rmse = compute_RMSE(ymu,y_test);

end

