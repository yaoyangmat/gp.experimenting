function con_bayesian_opt_demo()
% This demo runs bayesian optimisation on the "Forrester function"
% The two improvement criteria being compared are:
% Probablility of improvement (PI)
% Expectation of improvement (EI)

    % Set up basic parameters
    f = @(x) ((x.*6-2).^2).*sin((x.*6-2).*2);           % Target function 
    acq_type = 'EI';
    
    x_train = [0; 0.18; 0.25; 0.28; 0.5; 0.95; 1];
    y_train = f(x_train);
    n_test = 100;
    x_test = linspace(0,1,n_test)';
    y_test = f(x_test);
    fmin = min(y_test);
    
    % Train surrogates 
    gpdata = gp_metamodel('Train', x_train, y_train);
    
    % Constraint function
    pd1 = makedist('Normal',0.9,0.25);
    pd2 = makedist('Normal',0,0.45);
    f_constraint = @(x) 1.3 - pdf(pd1,x)- pdf(pd2,x);   
    c_train = f_constraint(x_train);
    c_test = f_constraint(x_test);
    c_gpdata = gp_metamodel('Train', x_train, c_train);

    % Set up plot
    fig = figure; 
    set(gcf, 'Units', 'normalized', 'Position', [0.5, 0.05, 0.5, 0.8])
    WinOnTop(fig,true);

    %% Continue drawing samples based on improvement criteria
    n_iters = 20;
    for i = 1:n_iters    
        % Get update based on improvement criteria
        [optimal_y,~] = min(y_train);
        [improvement,x_new] = get_update_1d( acq_type, gpdata, x_test, optimal_y );
        y_new = f(x_new);
        c_new = f_constraint(x_new);
        
        % Print results
        diff = optimal_y - fmin;
        fprintf('Iteration %d: Distance from true optimum: %.6f\n',i,diff)
        
        % Plot and request for new update
        update_plot(gpdata, x_test, y_test, improvement, x_new, c_gpdata, c_test)
        %update_plot(gpdata, x_test, y_test, improvement, x_new)
        prompt = 'Press any key to generate point, Quit (q) ';
        if strcmp(input(prompt,'s'),'q')
            close(fig); break;
        end
        clf(fig); % Delete old plots
        
        % Add to training set
        x_train = [x_train; x_new];
        y_train = [y_train; y_new];   

        % Update surrogate
        gpdata = gp_metamodel('Train', x_train, y_train);
        
        % Constraint function
        c_train = [c_train; c_new]; 
        c_gpdata = gp_metamodel('Train', x_train, c_train);
        
    end
end

%% Plotting functions
function update_plot(gpdata, x_test, y_test, improvement, x_new, c_gpdata, c_test)

    if nargin > 5
        n_subplots = 3;
    else
        n_subplots = 2;
    end
   
    subplot(n_subplots,1,1);
    update_gp_plot(gpdata, x_test, y_test, -10, 20)
    ylabel('Objective');
    vline(x_new);

    subplot(n_subplots,1,2)
    plot(x_test,improvement);
    ylabel('Improvement criteria');
    vline(x_new);
    
    if nargin > 5
        subplot(n_subplots,1,3)
        update_gp_plot(c_gpdata, x_test, c_test, -1, 2)
        ylabel('Constraint');
        vline(x_new);
        hline(0);
    end
end

function update_gp_plot(gpdata, x_test, y_test, y_min, y_max)
    scatter_sz = 30;
    
    % Make new prediction of underlying function
    [ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, x_test);
    ys = sqrt(ys2);

    scatter(gpdata.X, gpdata.Y+gpdata.offset, scatter_sz,'bo','LineWidth',1); ylim([y_min,y_max]); hold on; 
    plot(x_test,y_test); ylim([y_min,y_max]);
    plot(x_test,ymu); ylim([y_min,y_max]);
    jbfill(x_test',ymu'+2*ys',ymu'-2*ys','b','k',1,0.1); % Fill in uncertainty bounds
    legend('Training pts', 'True fn', 'Pred fn');
end

