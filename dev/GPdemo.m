lb = -5; ub = 5;
n_test = 100;
X_test = linspace(lb,ub,n_test);
s = 1e-6;

% Initialise GP prior -> set mean=0 and kernel parameters
mu = 0; 
ell = 1;
sf = 1; 
hyp = [ell,sf];

mu_test = ones(1,n_test)*mu;
s2_test = ones(1,n_test)*sf;
s_test = sqrt(s2_test);

fig = figure; hold on; 

plot(X_test,mu_test);
jbfill(X_test,mu_test+2*s_test,mu_test-2*s_test,'b','k',1,0.2);     % Fill in uncertainty bounds
ylim([-10,10]);

n_iters = 5;
X_train = [];
for i = 1:n_iters
    % Sample an input point, add to X_test
    x_new = rand()*(ub-lb) + lb;
    X_train = [X_train; x_new];
    
    % Retrain GP -> Compute K and L
    K = covSEbasic(X_train, X_train, hyp);
    L = chol(K + s*eye(length(X_train)));
    
    % Obtain output by sampling from predicted distribution
    foo = 1;
    
    
    % Plot
    line = vline(x_new);
    disp('Click to generate new point...'); waitforbuttonpress;
    delete(line);
end