clear all; clc;
lb = -5; ub = 5;
y_min = -4; y_max = 4;
n_test = 100;
x_test = linspace(lb,ub,n_test);
x_test = x_test';
s = 1e-6;

% Initialise GP prior -> set mean=0 and kernel parameters
mu = 0; 
ell = 1;            % default: 1
sn = 0.5;           % default: 0.5    
hyp_t = [ell;sn];
cov_t = @covSEbasic;

cov = {@covSEard}; 
logtheta0 = log([ell;sn]);
hyp.cov = logtheta0;  
emptymean = [];
lik = {@likGauss};    
sy = exp(-6);       % default: -4
hyp.lik = log(sy);       
inf = @infGaussLik;

mu_test = ones(n_test,1)*mu;
s2_test = ones(n_test,1)*sn.^2;
s_test = sqrt(s2_test);

fig = figure; ax = gca; 
pred_fn = plot(x_test,mu_test);
pred_bds = jbfill(x_test',mu_test'+2*s_test',mu_test'-2*s_test','b','k',1,0.2);     % Fill in uncertainty bounds

ylim([y_min,y_max]);
ylim manual;
hold on; 
WinOnTop(fig,true);

n_iters = 100;
x_train = [];
y_train = [];
for i = 1:n_iters
    % Pick an input point, x_star, at random
    x_star = rand()*(ub-lb) + lb;
    line = vline(x_star);
    
    prompt = 'Press any key to generate point, Quit (Q) ';
    x = input(prompt,'s');
    if strcmp(x,'q')
        close(fig)
        break
    end

    % Obtain mu_star and s_star
    if isempty(x_train)
        mu_star = 0; s2_star = 1;
    else
        [mu_star,s2_star] = gp(hyp,inf,emptymean,cov,lik,x_train,y_train,x_star);
        %[mu_star, s2_star] = gp_predict(hyp_t, cov_t, x_train, y_train, x_star);
    end
    
    % Sample y_star from predicted distribution
    s_star = sqrt(s2_star);
    y_star = mu_star + s_star * randn();
    
    % Add to training set
    x_train = [x_train; x_star];
    y_train = [y_train; y_star];   
    
    % Retrain predicted function on X_test
    [mu_test, s2_test] = gp(hyp,inf,emptymean,cov,lik,x_train,y_train,x_test);
    %[mu_test, s2_test] = gp_predict(hyp_t, cov_t, x_train, y_train, x_test);
    s_test = sqrt(s2_test);
    
    % Plot results
    delete(line);
    delete(pred_fn);
    delete(pred_bds);
    scatter(ax, x_train, y_train,'x','r'); ylim([y_min,y_max]); hold on; 
    pred_fn = plot(x_test,mu_test); ylim([y_min,y_max]);
    pred_bds = jbfill(x_test',mu_test'+2*s_test',mu_test'-2*s_test','b','k',1,0.2);  ylim([y_min,y_max]);   % Fill in uncertainty bounds
    
end