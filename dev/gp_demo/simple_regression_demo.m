% Attempt to fit a simple sin curve

%% Basic parameters
MAX_NUM_EVAL = 50;              % Maximum allowed function evals
n_train = 8;                   % Number of training points
n_test = 100;                   % Number of test points

%% Setting up data - training and test
x_train = linspace(-5,5,n_train)';
y_train = sin(x_train);

x_test = linspace(-7, 7, n_test)';
y_test = sin(x_test);

%% Initialize hyp, cov, mean, lik
% Initialise guess: logtheta0 (from Swordfish)
stdX = std(x_train)';
stdX( stdX./abs(mean(x_train))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(y_train); 0.05*std(y_train)]);

hyp_t = logtheta0;  
cov_t = @kernel;

% Using GPML
cov = {@covSum, {@covSEard,@covNoise}};
lik = {@likGauss};  
inf = @infGaussLik;

hyp.cov = logtheta0;
hyp.lik = logtheta0(end);   

% Learn hyperparameters
hyp_t = minimize(hyp_t,@gp_train,-MAX_NUM_EVAL,cov_t,x_train,y_train); 
hyp = minimize(hyp, @gp, -MAX_NUM_EVAL, inf, [], cov, lik, x_train, y_train);

% Predictions
[ ymu, ys2 ] = gp_predict( hyp_t,cov_t,x_train,y_train,x_test );
%[ ymu, ys2 ] = gp(hyp, inf, [], cov, lik, x_train, y_train, x_test);
ys = sqrt(ys2);

% Plot
figure;
hold on;
plot(x_test,y_test);
plot(x_test,ymu);
scatter(x_train,y_train);
jbfill(x_test',ymu'+2*ys',ymu'-2*ys','b','k',1,0.2); % Fill in uncertainty bounds
legend('True fn', 'Pred fn', 'Training pts');
hold off;
