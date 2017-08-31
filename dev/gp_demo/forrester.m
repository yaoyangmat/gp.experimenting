%f = @(x) ((x.*6-2).^2).*sin((x.*6-2).*2);

pd1 = makedist('Normal',0.9,0.25);
pd2 = makedist('Normal',0,0.45);
f_constraint = @(x) 1.3 - pdf(pd1,x)- pdf(pd2,x);
y_min = -2; 
y_max = 2;

% Set up train and test points
x_train = [0; 0.18; 0.25; 0.28; 0.5; 0.95; 1];
y_train = f_constraint(x_train);
n_test = 100;
x_test = linspace(0,1,n_test)';
y_test = f_constraint(x_test);

% Train and predict
gpdata = gp_metamodel('Train', x_train, y_train);
[ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, x_test);
ys = sqrt(ys2);

% Set up plot
fig = figure; scatter_sz = 100;
set(gcf, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.5, 0.8])
training_pts = scatter(x_train,y_train,scatter_sz,'bx'); ylim([y_min,y_max]); hold on; 
true_fn = plot(x_test,y_test); ylim([y_min,y_max]);
pred_fn = plot(x_test,ymu); ylim([y_min,y_max]);
pred_bds = jbfill(x_test',ymu'+2*ys',ymu'-2*ys','b','k',1,0.2); % Fill in uncertainty bounds

legend('Training pts', 'True fn', 'Pred fn');
