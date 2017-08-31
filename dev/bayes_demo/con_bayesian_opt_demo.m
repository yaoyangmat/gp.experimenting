function con_bayesian_opt_demo()
% This demo runs bayesian optimisation on the "Forrester function"
% The two improvement criteria being compared are:
% Probablility of improvement (PI)
% Expectation of improvement (EI)

    % Set up basic parameters
    f = forrester_fn;                       % Target function 
    f_con = forrester_constraint_fn;        % Constraint function
    c_lim = [0]; 
    c_type = ['>'];
    
    objectives = { f };
    constraints = { f_con };
    
    acq_type = 'EI';
    
    x_train = [0; 0.18; 0.25; 0.28; 0.5; 0.95; 1];
    x_test = linspace(0,1,100)';

    y_train = get_response(objectives, x_train);
    y_test = get_response(objectives, x_test);
    c_train = get_response(constraints, x_train);
    c_test = get_response(constraints, x_test);
    
    [ x_min, y_min, c_num ] = get_optimal( x_test, y_test, c_test, c_lim, c_type );
    
    % fmin = get_optimal_result(y_train, c_train);
    fmin = min(y_test);
    
    % Train surrogates 
    gpdata = gp_metamodel('Train', x_train, y_train);
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
        c_new = f_con(x_new);
        
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

