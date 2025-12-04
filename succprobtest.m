%%%COMPUTE SUCCESS PROBABILITY FOR SOME HILBERT SPACE TRUNCATION LOG2(n)%%%
N=2^10;
xlist=linspace(-N/2, N/2-1, N)';
wlist=sin(2*xlist/N);

W = 6;
targ= @(x) exp(-(W^2)*(asin(x).^2)/2);

%%% REMEZ SOLVER CODE %%%
deg = 22;
opts.intervals=[0,sin(1)];
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=false;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value22 = func(wlist);

%%% POLY MAX OVER INTERVAL, FNCS TO COMPUTE PROBABILITY%%%
fcnmax=0.989987;
norm=sqrt(2*W*sum(func_value22.^2)/N);
l2normfilling=norm/sqrt(2*W*(fcnmax)^2);

%%%PLOT USING SUCCESS PROBABILITIES COMPUTED FROM THE CODE ABOVE%%%
nlist=2:10;
normlist=[1.7149, 1.3343,  1.3180, 1.3180, 1.3180, 1.3180,1.3180, 1.3180, 1.3180];
l2normfillinglist=[0.5001, 0.3891, 0.3843,  0.3843, 0.3843, 0.3843, 0.3843, 0.3843, 0.3843];
problist=(0.5*l2normfillinglist).^2;
figure()
xlabel('$m\in[2, 10]$', 'Interpreter', 'latex')
ylabel('$f_\mathrm{poly}(x)$', 'Interpreter', 'latex')

plot(nlist,problist);
print(gcf,'gaussian_state_preparation.png','-dpng','-r500');
matlab2tikz('succprobplot.tex')
