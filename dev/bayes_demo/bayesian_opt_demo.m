function bayesian_opt_demo(is_constrained)
% This demo runs bayesian optimisation on the "Forrester function"
% If the Forrester constraint is used, then 
% the acquisition function = improvement criteria * P(feasibility)
%
% The two improvement criteria that can be compared are:
% Probablility of improvement (PI) / constrained PIPF
% Expectation of improvement (EI) / constrained EIPF

    % Target function 
    f = forrester_fn;                      
    objectives = { f };
    
    % Constraint function (if any)
    if is_constrained
        f_con = forrester_constraint_fn;        
        c_lim = [0]; 
        c_type = ['>'];
        constraints = { f_con };
    else                           
        c_lim = []; 
        c_type = [];
        constraints = { };
        c_gpdata = [];
    end
    
    % Acquisition function can be EI or PI
    acq_type = 'EI';
    
    % Compute train and test points
    x_train = [0; 0.18; 0.25; 0.28; 0.5; 0.95; 1];
    x_test = linspace(0,1,100)';

    y_train = get_response(objectives, x_train);
    y_test = get_response(objectives, x_test);
    c_train = get_response(constraints, x_train);
    c_test = get_response(constraints, x_test);
    
    % Obtain the true optimal based on test points
    [ x_optimal, y_optimal, ~ ] = get_optimal( x_test, y_test, c_test, c_lim, c_type );
    
    % Train surrogates 
    gpdata = gp_metamodel('Train', x_train, y_train);
    for i = 1:length(c_lim)
        c_gpdata(i) = gp_metamodel('Train', x_train, c_train(:,i));
    end
    
    % Set up plot
    fig = figure; 
    set(gcf, 'Units', 'normalized', 'Position', [0.5, 0.05, 0.5, 0.8])
    WinOnTop(fig,true);

    %% Continue drawing samples based on improvement criteria
    n_iters = 20;
    for i = 1:n_iters    
        % Get update based on improvement criteria
        [ ~, f_min, ~ ] = get_optimal( x_train, y_train, c_train, c_lim, c_type );
        [ acq,x_new ] = get_update_1d( acq_type, gpdata, x_test, f_min, c_gpdata, c_lim, c_type );
        y_new = get_response(objectives, x_new);
        c_new = get_response(constraints, x_new);
        
        % Print results
        diff = abs(y_optimal - f_min);
        fprintf('Iteration %d: Distance from true optimum: %.6f\n',i,diff)
        
        % Plot and request for new update
        if is_constrained
            update_plot(gpdata, x_test, y_test, acq, x_optimal, x_new, c_gpdata, c_test);
        else
            update_plot(gpdata, x_test, y_test, acq, x_optimal, x_new);
        end

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
        
        % Handle constraints (if any)
        c_train = [c_train; c_new]; 
        for i = 1:length(c_lim)
            c_gpdata(i) = gp_metamodel('Train', x_train, c_train(:,i));
        end
        
    end
end

%% Plotting functions
function update_plot(gpdata, x_test, y_test, acq, x_optimal, x_new, c_gpdata, c_test)

    if nargin > 6
        n_subplots = 3;
    else
        n_subplots = 2;
    end
   
    subplot(n_subplots,1,1);
    update_gp_plot(gpdata, x_test, y_test, -10, 20)
    ylabel('Objective function');
    vline(x_new);
    vline(x_optimal, 'm', 'True optimal');

    subplot(n_subplots,1,2)
    plot(x_test,acq);
    ylabel('Acquisition function');
    vline(x_new);
    vline(x_optimal, 'm', 'True optimal');
    
    if n_subplots == 3;
        subplot(n_subplots,1,3)
        update_gp_plot(c_gpdata, x_test, c_test, -1, 1.5)
        ylabel('Constraint');
        vline(x_new);
        vline(x_optimal, 'm', 'True optimal');
        hline(0,'k','Constraint');
    end
end

function update_gp_plot(gpdata, x_test, y_test, y_min, y_max)
    scatter_sz = 30;
    
    % Make new prediction of target function
    [ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, x_test);
    ys = sqrt(ys2);

    scatter(gpdata.X, gpdata.Y+gpdata.offset, scatter_sz,'bo','LineWidth',1); ylim([y_min,y_max]); hold on; 
    plot(x_test,y_test); ylim([y_min,y_max]);
    plot(x_test,ymu); ylim([y_min,y_max]);
    jbfill(x_test',ymu'+2*ys',ymu'-2*ys','b','k',1,0.1); % Fill in uncertainty bounds
    legend('Training pts', 'True fn', 'Pred fn');
end

