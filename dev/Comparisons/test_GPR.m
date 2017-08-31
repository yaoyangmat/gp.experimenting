f = @(x) ((x.*6-2).^2).*sin((x.*6-2).*2);
y_min = -10; 
y_max = 20;

% Set up train and test points
x_train = [0; 0.18; 0.25; 0.28; 0.5; 0.95; 1];
y_train = f(x_train);
n_test = 10000;
x_test = linspace(0,1,n_test)';
y_test = f(x_test);

gpdata = gaussianprocessregression('Train', x_train, y_train);
[ ymu, ys2 ] = gaussianprocessregression('Evaluate', x_test, gpdata);

gpdata.X = gpdata.x;
[ ymu, ys2 ] = gp_metamodel('Evaluate', gpdata, x_test);
foo = 1;