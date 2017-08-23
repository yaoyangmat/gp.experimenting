% I downloaded the data from
%   ftp://ftp.cmdl.noaa.gov/ccg/co2/trends/co2_mm_mlo.txt
% and pre-processed using
%   tail -n 637 co2_mm_mlo.txt | awk '{ print $3, " ", $4 }' > mauna.txt
% Carl Edward Rasmussen, 2011-04-20.

load mauna.txt
z = mauna(:,2) ~= -99.99;                             % get rid of missing data
year = mauna(z,1); co2 = mauna(z,2);       % extract year and CO2 concentration

x = year(year<2004); y = co2(year<2004);                        % training data
xx = year(year>2004); yy = co2(year>2004);                          % test data

k1 = @covSEiso;                     % covariance contributions, long term trend
k2 = {@covProd, {@covPeriodic, @covSEisoU}};      % close to periodic component
k3 = @covRQiso;                      % fluctations with different length-scales
k4 = @covSEiso;                 % very short term (month to month) correlations 
covfunc = {@covSum, {k1, k2, k3, k4}};                % add up covariance terms

meanfunc = {@meanSum, {@meanLinear, @meanConst}};

hyp.cov = [4 4 0 0 1 4 0 0 -1 -2 -2]; 
hyp.lik = -2;
hyp.mean = [1.5 -2630]

[hyp fX i] = ...                                             % fit the GP model
     minimize(hyp, @gp, -500, @infExact, meanfunc, covfunc, @likGauss, x, y);

zz = (2004+1/24:1/12:2024-1/24)';   % make predictions 20 years into the future
[mu s2] = gp(hyp, @infExact, meanfunc, covfunc, @likGauss, x, y, zz);

f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([zz; flipdim(zz,1)], f, [7 7 7]/8); hold on;           % show predictions
plot(x,y,'b.'); plot(xx,yy,'r.')                               % with the data
