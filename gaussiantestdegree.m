%%%ADAPTED FROM QSPPACK GAUSSIAN EXAMPLE%%%

W = 6;
targ= @(x) exp(-(W^2)*(asin(x).^2)/2);
targunwound= @(x) exp(-((x).^2)/2);

xunwoundlist = linspace(-W,W,500)';
wlist=sin(xunwoundlist/W);

targ_value = targ(wlist);

% To numerically find the best even polynomial approximating $h(z)$,  we use
% a subroutine which solves the problem using convex optimization. Here are
% the parameters of the subroutine.



deg = 8;
opts.intervals=wlist;
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value8 = func(wlist)


deg = 12;
opts.intervals=[0,sin(1)];
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value12 = func(wlist);


deg = 16;
opts.intervals=[0,sin(1)];
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value16 = func(wlist);

deg = 20;
opts.intervals=wlist;
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value20 = func(wlist);

deg = 22;
opts.intervals=wlist;
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value22 = func(wlist);

deg = 24;
opts.intervals=wlist;
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value24 = func(wlist);

deg = 26;
opts.intervals=wlist;
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value26 = func(wlist);

deg = 28;
opts.intervals=wlist;
opts.objnorm = Inf;
opts.epsil = 0.01;
opts.npts = 500;
opts.fscale = 0.99;
opts.isplot=true;

coef_full=cvx_poly_coef(targ, deg, opts);
parity = mod(deg, 2);
coef = coef_full(1+parity:2:end);
func = @(y) ChebyCoef2Func(y, coef, parity, true);
func_value28 = func(wlist);

figure()
xlabel('$x\in[-W, W]$', 'Interpreter', 'latex')
ylabel('$f_\mathrm{poly}(x)$', 'Interpreter', 'latex')
plot(xunwoundlist, targunwound(xunwoundlist))
hold on
plot(xunwoundlist,func_value8);
hold on
plot(xunwoundlist,func_value12);
hold on
plot(xunwoundlist,func_value16);
hold on
plot(xunwoundlist,func_value20);
hold on
plot(xunwoundlist,func_value22);
% hold on
% plot(xunwoundlist,func_value24);
% hold on
% plot(xunwoundlist,func_value26);
% hold on
% plot(xunwoundlist,func_value28);
legend('$f(x)$','$p_{8}(y)$', '$p_{12}(y)$', '$p_{16}(y)$', '$p_{20}(y)$', '$p_{22}(y)$')
% legend('$f(x)$','$p_{20}(y)$', '$p_{22}(y)$', '$p_{24}(y)$', '$p_{26}(y)$', '$p_{28}(y)$')
% legend('$f(x)$','$p_{22}(y)$', '$p_{24}(y)$', '$p_{26}(y)$', '$p_{28}(y)$')
print(gcf,'gaussian_state_preparation.png','-dpng','-r500');
matlab2tikz('gaussalpha0degreerange.tex')

figure()
xlabel('$x\in[-W, W]$', 'Interpreter', 'latex');
ylabel('$\left|f(x)-p_{d}(x)$');

plot(xunwoundlist,targunwound(xunwoundlist)-func_value20);
hold on
plot(xunwoundlist,targunwound(xunwoundlist)-func_value22);
hold on
plot(xunwoundlist,targunwound(xunwoundlist)-func_value24);
hold on
plot(xunwoundlist,targunwound(xunwoundlist)-func_value26);
hold on
plot(xunwoundlist,targunwound(xunwoundlist)-func_value28);

legend('$p_{20}(y)$', '$p_{22}(y)$', '$p_{24}(y)$', '$p_{26}(y)$', '$p_{28}(y)$')
matlab2tikz('gaussalpha0errorrange.tex')

