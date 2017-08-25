clear all;clc;
f = @(x) ((x.*6-2).^2).*sin((x.*6-2).*2);
y_min = -10; 
y_max = 20;

% Set up train and test points
x_train = [0; 0.18; 0.25; 0.28; 0.5; 0.95; 1];
y_train = f(x_train);
n_test = 100;
x_test = linspace(0,1,n_test)';
y_test = f(x_test);

% Train and predict
gpdata = gp_metamodel('Train', x_train, y_train);
[ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, x_test);
ys = sqrt(ys2);

% Calculate improvement criteria
[optimal_y,idx] = min(y_train);
p_improvement = get_improvement_criteria( 'P_improvement', gpdata, x_test, optimal_y );
e_improvement = get_improvement_criteria( 'E_improvement', gpdata, x_test, optimal_y );

% Set up plot
fig = figure; 
set(gcf, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.5, 0.8])
subplot(3,1,1); ax = gca; 
training_pts = scatter(x_train,y_train,'bx'); ylim([y_min,y_max]); hold on; 
true_fn = plot(x_test,y_test); ylim([y_min,y_max]);
pred_fn = plot(x_test,ymu); ylim([y_min,y_max]);
pred_bds = jbfill(x_test',ymu'+2*ys',ymu'-2*ys','b','k',1,0.2); % Fill in uncertainty bounds

legend('Training pts', 'True fn', 'Pred fn');

subplot(3,1,2)
p_imp = plot(x_test,p_improvement);
ylabel('P(improvement)');

subplot(3,1,3)
e_imp = plot(x_test,e_improvement);
ylabel('E(improvement)');

WinOnTop(fig,true);

%% Continue drawing samples based on expected improvement
n_iters = 100;
for i = 1:n_iters    
    
    % Pick a new input point, x_new, based on the maximum improvement
    [optimal_i,idx] = max(p_improvement);
    x_new = x_test(idx);
    y_new = f(x_new);
    subplot(3,1,1); line = vline(x_new);
    
    prompt = 'Press any key to generate point, Quit (Q) ';
    x = input(prompt,'s');
    if strcmp(x,'q')
        close(fig)
        break
    end

    % Add to training set
    x_train = [x_train; x_new];
    y_train = [y_train; y_new];   
    
    % Train and predict
    gpdata = gp_metamodel('Train', x_train, y_train);
    [ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, x_test);
    ys = sqrt(ys2);
    
    % Calculate improvement criteria
    [optimal_y,idx] = min(y_train);
    p_improvement = get_improvement_criteria( 'P_improvement', gpdata, x_test, optimal_y );
    e_improvement = get_improvement_criteria( 'E_improvement', gpdata, x_test, optimal_y );

    % Plot results
    subplot(3,1,1); 
    delete(line);
    delete(pred_fn);
    delete(pred_bds); 
    training_pts = scatter(x_train,y_train,'bx'); ylim([y_min,y_max]); hold on; 
    true_fn = plot(x_test,y_test); ylim([y_min,y_max]);
    pred_fn = plot(x_test,ymu); ylim([y_min,y_max]);
    pred_bds = jbfill(x_test',ymu'+2*ys',ymu'-2*ys','b','k',1,0.2); % Fill in uncertainty bounds
    
    subplot(3,1,2); 
    delete(p_imp);
    p_imp = plot(x_test,p_improvement);
    
    subplot(3,1,3);
    delete(e_imp);
    e_imp = plot(x_test,e_improvement);
   
end