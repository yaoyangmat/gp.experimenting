function varargout = gp_metamodel(Action, varargin)
%METAMODEL 
    
%   gpdata = gp_metamodel('Train',X,Y)
%   [ymu,ys2] = gp_metamodel('Evaluate',gpdata,Xs)

switch Action
    case 'Train'
        X = varargin{1};
        Y = varargin{2};
        MAX_NUM_EVAL = 50; 
        
        % Demean the data
        offset = mean(Y);
        Y = Y - offset;

        % Initialise guess: logtheta0 (from Swordfish)
        stdX = std(X)';
        stdX( stdX./abs(mean(X))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
        logtheta0 = log([stdX; std(Y); 0.05*std(Y)]);
        
        logtheta0(end) = -5;
        
        hyp = logtheta0;  
        cov = @kernel; 

        hyp = minimize(hyp,@gp_train,-MAX_NUM_EVAL,cov,X,Y); 
        
        % Precompute L and alpha
        K = cov(hyp,X);
        L = chol(K)'; 
        alpha = L'\(L\Y);
        
        gpdata.offset = offset;
        gpdata.hyp = hyp;
        gpdata.cov = cov;
        gpdata.X = X;
        gpdata.Y = Y;
        gpdata.L = L;
        gpdata.alpha = alpha;
        varargout{1} = gpdata;
        
    case 'Evaluate'
        gpdata = varargin{1};  
        Xs = varargin{2}; 

        L = gpdata.L;
        alpha = gpdata.alpha;
        
        [Kss, Kstar] = gpdata.cov(gpdata.hyp, gpdata.X, Xs);
        Lk = L\Kstar;
        ymu = Kstar' * alpha;
        ys2 = Kss - sum(Lk.^2, 1)';

        ymu = ymu + gpdata.offset;
        varargout{1} = ymu;
        varargout{2} = ys2;
end

