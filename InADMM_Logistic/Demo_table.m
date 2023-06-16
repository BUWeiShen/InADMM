rand('seed', 0);
randn('seed', 0);
%% table header
caption=' \caption{Elastic-net Logistic Regression ($a=0.5$) }';
label='\label{tab:EN_Logistic_5}';
begin_adjustment='\begin{adjustbox}{center,max width=1\textwidth}';
scale_box='\scalebox{0.95}{';
begin_tabular='\begin{tabular}{|c|l|l|l|l|l|l|}';
headerline='$(n,p,sd) $  & Algorithm & Iteration & Sum CG & Mean/Max CG & Time & Obj \\ \hline';
latex = {'\begin{table}[]';caption;label;begin_adjustment;scale_box;begin_tabular;'\hline';headerline};
%% case 1
columntitle1='\multirow{10}{*}{$(1*10^4,2*10^5,0.01\%)$}';
latex = [latex;columntitle1];
n = 10000;
p = 4*1e+5;
w = sprandn(p, 1, 0.1);  % N(0,1), 10% sparse
v = randn(1);            % random intercept
X = sprandn(n, p, 0.01/100);
btrue = sign(X*w + v);
b = sign(X*w + v + sqrt(0.1)*randn(n,1)); % labels with noise
A = spdiags(b, 0, n, n) * X;
nu=eigs(A'*A,1);
ratio = sum(b == 1)/(n);
x_true = [v; w];

C=n;
rho_inner=1/2;
rho_outer=1/2;
lam_max=norm((1-ratio)*sum(A(b==1,:),1) + ratio*sum(A(b==-1,:),1), 'inf');
lambda=0.1*lam_max;
l1_ratio=0.5;
lambda_1=l1_ratio*lambda;
lambda_2=(1-l1_ratio)*lambda;
mu_outer=sqrt(lambda_2);
mu_inner=sqrt(lambda_2);
max_time=1000;
MAX_ITER=500;
ABSTOL   = 1e-4;
RELTOL   = 1e-3;



% parameters for input data
param.A          = A;
param.b          = b;

% parameters for elastice logistic model
param.C          = C;
param.lambda_1   = lambda_1;
param.lambda_2   = lambda_2;

% parameters for beta-step solver
param.rho_outer  = rho_outer;
param.rho_inner  = rho_inner;
param.mu_outer   = mu_outer;
param.mu_inner   = mu_inner;
param.nu         = nu;

param.MAX_ITER = MAX_ITER;
param.ABSTOL   = ABSTOL;
param.RELTOL   = RELTOL;
param.max_time=max_time;




%inner: GD
withobj=0;
param.withobj=withobj;
inSolver='GD';
param.inSolver=inSolver;
[~, history_GD] = InADMM_EN_Logistic(param);
obj_GD=history_GD.objval(end);
length_GD=length(history_GD.iters);
sumiter_GD=sum(history_GD.iters);
mean_GD=mean(history_GD.iters);
max_GD=max(history_GD.iters);
time_GD=max(history_GD.time);
param.obj=obj_GD;

 



if length_GD<500 && time_GD<1000
    input=['&$\textbf{GD}$  &', num2str(length_GD), '&', num2str(sumiter_GD), '&', num2str(mean_GD), '/' ,num2str(max_GD),  '&' ,num2str(time_GD), '&' ,num2str(obj_GD),   '\\ '];
 elseif length_GD>=500
    input=['&$\textbf{GD}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_GD>=1000
     input=['&$\textbf{GD}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];



%inner: GD_fix_tol 10
withobj=1;
param.withobj=withobj;
inSolver='GD_fix_tol';
param.inSolver=inSolver;
param.tol_fix=10;
[~, history_GD_fix10] = InADMM_EN_Logistic(param);

obj_GD_fix10=history_GD_fix10.objval(end);
length_GD_fix10=length(history_GD_fix10.iters);
sumiter_GD_fix10=sum(history_GD_fix10.iters);
mean_GD_fix10=mean(history_GD_fix10.iters);
max_GD_fix10=max(history_GD_fix10.iters);
time_GD_fix10=max(history_GD_fix10.time);


 if length_GD_fix10<500 && time_GD_fix10<1000
    input=['&$\text{GD}_{1e+1}$  &', num2str(length_GD_fix10), '&', num2str(sumiter_GD_fix10), '&', num2str(mean_GD_fix10), '/' ,num2str(max_GD_fix10),  '&' ,num2str(time_GD_fix10), '&' ,num2str(obj_GD_fix10),   '\\ '];
 elseif length_GD_fix10>=500
    input=['&$\text{GD}_{1e+1}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_GD_fix10>=1000
     input=['&$\text{GD}_{1e+1}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];

%inner: GD_fix_tol 1
withobj=1;
param.withobj=withobj;
inSolver='GD_fix_tol';
param.inSolver=inSolver;
param.tol_fix=1;
[~, history_GD_fix1] = InADMM_EN_Logistic(param);

obj_GD_fix1=history_GD_fix1.objval(end);
length_GD_fix1=length(history_GD_fix1.iters);
sumiter_GD_fix1=sum(history_GD_fix1.iters);
mean_GD_fix1=mean(history_GD_fix1.iters);
max_GD_fix1=max(history_GD_fix1.iters);
time_GD_fix1=max(history_GD_fix1.time);


 if length_GD_fix1<500 && time_GD_fix1<1000
    input=['&$\text{GD}_{1e+0}$  &', num2str(length_GD_fix1), '&', num2str(sumiter_GD_fix1), '&', num2str(mean_GD_fix1), '/' ,num2str(max_GD_fix1),  '&' ,num2str(time_GD_fix1), '&' ,num2str(obj_GD_fix1),   '\\ '];
 elseif length_GD_fix1>=500
    input=['&$\text{GD}_{1e+0}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_GD_fix1>=1000
     input=['&$\text{GD}_{1e+0}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];


%inner: GD_fix_tol 0.1
withobj=1;
param.withobj=withobj;
inSolver='GD_fix_tol';
param.inSolver=inSolver;
param.tol_fix=0.1;
[~, history_GD_fixPiont1] = InADMM_EN_Logistic(param);

obj_GD_fixPiont1=history_GD_fixPiont1.objval(end);
length_GD_fixPiont1=length(history_GD_fixPiont1.iters);
sumiter_GD_fixPiont1=sum(history_GD_fixPiont1.iters);
mean_GD_fixPiont1=mean(history_GD_fixPiont1.iters);
max_GD_fixPiont1=max(history_GD_fixPiont1.iters);
time_GD_fixPiont1=max(history_GD_fixPiont1.time);


 if length_GD_fixPiont1<500 && time_GD_fixPiont1<1000
    input=['&$\text{GD}_{1e-1}$  &', num2str(length_GD_fixPiont1), '&', num2str(sumiter_GD_fixPiont1), '&', num2str(mean_GD_fixPiont1), '/' ,num2str(max_GD_fixPiont1),  '&' ,num2str(time_GD_fixPiont1), '&' ,num2str(obj_GD_fixPiont1),   '\\ '];
 elseif length_GD_fixPiont1>=500
    input=['&$\text{GD}_{1e-1}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_GD_fixPiont1>=1000
     input=['&$\text{GD}_{1e-1}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];

%inner: GD_fix_tol 0.01
withobj=1;
param.withobj=withobj;
inSolver='GD_fix_tol';
param.inSolver=inSolver;
param.tol_fix=0.01;
[~, history_GD_fixPiont2] = InADMM_EN_Logistic(param);

obj_GD_fixPiont2=history_GD_fixPiont2.objval(end);
length_GD_fixPiont2=length(history_GD_fixPiont2.iters);
sumiter_GD_fixPiont2=sum(history_GD_fixPiont2.iters);
mean_GD_fixPiont2=mean(history_GD_fixPiont2.iters);
max_GD_fixPiont2=max(history_GD_fixPiont2.iters);
time_GD_fixPiont2=max(history_GD_fixPiont2.time);


 if length_GD_fixPiont2<500 && time_GD_fixPiont2<1000
    input=['&$\text{GD}_{1e-2}$  &', num2str(length_GD_fixPiont2), '&', num2str(sumiter_GD_fixPiont2), '&', num2str(mean_GD_fixPiont2), '/' ,num2str(max_GD_fixPiont2),  '&' ,num2str(time_GD_fixPiont2), '&' ,num2str(obj_GD_fixPiont2),   '\\ \cline{2-7}'];
 elseif length_GD_fixPiont2>=500
    input=['&$\text{GD}_{1e-2}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ \cline{2-7}'];
 elseif time_GD_fixPiont2>=1000
     input=['&$\text{GD}_{1e-2}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ \cline{2-7} '];
 end
latex=[latex;input];




%inner: InADMM
withobj=1;
param.withobj=withobj;
inSolver='InADMM';
param.inSolver=inSolver;
[~, history_InADMM] = InADMM_EN_Logistic(param);
obj_InADMM=history_InADMM.objval(end);
length_InADMM=length(history_InADMM.iters);
sumiter_InADMM=sum(history_InADMM.iters);
mean_InADMM=mean(history_InADMM.iters);
max_InADMM=max(history_InADMM.iters);
time_InADMM=max(history_InADMM.time);


 if length_InADMM<500 && time_InADMM<1000
    input=['&$\textbf{InADMM}$  &', num2str(length_InADMM), '&', num2str(sumiter_InADMM), '&', num2str(mean_InADMM), '/' ,num2str(max_InADMM),  '&' ,num2str(time_InADMM), '&' ,num2str(obj_InADMM),   '\\ '];
 elseif length_InADMM>=500
    input=['&$\textbf{InADMM}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_InADMM>=1000
     input=['&$\textbf{InADMM}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];



%inner: InADMM_fix_tol 10
withobj=1;
param.withobj=withobj;
inSolver='InADMM_fix_tol';
param.inSolver=inSolver;
param.tol_fix=10;
[~, history_InADMM_fix10] = InADMM_EN_Logistic(param);

obj_InADMM_fix10=history_InADMM_fix10.objval(end);
length_InADMM_fix10=length(history_InADMM_fix10.iters);
sumiter_InADMM_fix10=sum(history_InADMM_fix10.iters);
mean_InADMM_fix10=mean(history_InADMM_fix10.iters);
max_InADMM_fix10=max(history_InADMM_fix10.iters);
time_InADMM_fix10=max(history_InADMM_fix10.time);


 if length_InADMM_fix10<500 && time_InADMM_fix10<1000
    input=['&$\text{InADMM}_{1e+1}$  &', num2str(length_InADMM_fix10), '&', num2str(sumiter_InADMM_fix10), '&', num2str(mean_InADMM_fix10), '/' ,num2str(max_InADMM_fix10),  '&' ,num2str(time_InADMM_fix10), '&' ,num2str(obj_InADMM_fix10),   '\\ '];
 elseif length_InADMM_fix10>=500
    input=['&$\text{InADMM}_{1e+1}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_InADMM_fix10>=1000
     input=['&$\text{InADMM}_{1e+1}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];

%inner: InADMM_fix_tol 1
withobj=1;
param.withobj=withobj;
inSolver='InADMM_fix_tol';
param.inSolver=inSolver;
param.tol_fix=1;
[~, history_InADMM_fix1] = InADMM_EN_Logistic(param);

obj_InADMM_fix1=history_InADMM_fix1.objval(end);
length_InADMM_fix1=length(history_InADMM_fix1.iters);
sumiter_InADMM_fix1=sum(history_InADMM_fix1.iters);
mean_InADMM_fix1=mean(history_InADMM_fix1.iters);
max_InADMM_fix1=max(history_InADMM_fix1.iters);
time_InADMM_fix1=max(history_InADMM_fix1.time);


 if length_InADMM_fix1<500 && time_InADMM_fix1<1000
    input=['&$\text{InADMM}_{1e+0}$  &', num2str(length_InADMM_fix1), '&', num2str(sumiter_InADMM_fix1), '&', num2str(mean_InADMM_fix1), '/' ,num2str(max_InADMM_fix1),  '&' ,num2str(time_InADMM_fix1), '&' ,num2str(obj_InADMM_fix1),   '\\ '];
 elseif length_InADMM_fix1>=500
    input=['&$\text{InADMM}_{1e+0}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_InADMM_fix1>=1000
     input=['&$\text{InADMM}_{1e+0}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];


%inner: InADMM_fix_tol 0.1
withobj=1;
param.withobj=withobj;
inSolver='InADMM_fix_tol';
param.inSolver=inSolver;
param.tol_fix=0.1;
[~, history_InADMM_fixPiont1] = InADMM_EN_Logistic(param);

obj_InADMM_fixPiont1=history_InADMM_fixPiont1.objval(end);
length_InADMM_fixPiont1=length(history_InADMM_fixPiont1.iters);
sumiter_InADMM_fixPiont1=sum(history_InADMM_fixPiont1.iters);
mean_InADMM_fixPiont1=mean(history_InADMM_fixPiont1.iters);
max_InADMM_fixPiont1=max(history_InADMM_fixPiont1.iters);
time_InADMM_fixPiont1=max(history_InADMM_fixPiont1.time);


 if length_InADMM_fixPiont1<500 && time_InADMM_fixPiont1<1000
    input=['&$\text{InADMM}_{1e-1}$  &', num2str(length_InADMM_fixPiont1), '&', num2str(sumiter_InADMM_fixPiont1), '&', num2str(mean_InADMM_fixPiont1), '/' ,num2str(max_InADMM_fixPiont1),  '&' ,num2str(time_InADMM_fixPiont1), '&' ,num2str(obj_InADMM_fixPiont1),   '\\ '];
 elseif length_InADMM_fixPiont1>=500
    input=['&$\text{InADMM}_{1e-1}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_InADMM_fixPiont1>=1000
     input=['&$\text{InADMM}_{1e-1}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ '];
 end
latex=[latex;input];

%inner: InADMM_fix_tol 0.01
withobj=1;
param.withobj=withobj;
inSolver='InADMM_fix_tol';
param.inSolver=inSolver;
param.tol_fix=0.01;
[~, history_InADMM_fixPiont2] = InADMM_EN_Logistic(param);

obj_InADMM_fixPiont2=history_InADMM_fixPiont2.objval(end);
length_InADMM_fixPiont2=length(history_InADMM_fixPiont2.iters);
sumiter_InADMM_fixPiont2=sum(history_InADMM_fixPiont2.iters);
mean_InADMM_fixPiont2=mean(history_InADMM_fixPiont2.iters);
max_InADMM_fixPiont2=max(history_InADMM_fixPiont2.iters);
time_InADMM_fixPiont2=max(history_InADMM_fixPiont2.time);


 if length_InADMM_fixPiont2<500 && time_InADMM_fixPiont2<1000
    input=['&$\text{InADMM}_{1e-2}$  &', num2str(length_InADMM_fixPiont2), '&', num2str(sumiter_InADMM_fixPiont2), '&', num2str(mean_InADMM_fixPiont2), '/' ,num2str(max_InADMM_fixPiont2),  '&' ,num2str(time_InADMM_fixPiont2), '&' ,num2str(obj_InADMM_fixPiont2),   '\\ \hline'];
 elseif length_InADMM_fixPiont2>=500
    input=['&$\text{InADMM}_{1e-2}$  &', '>500', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ \hline'];
 elseif time_InADMM_fixPiont2>=1000
     input=['&$\text{InADMM}_{1e-2}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'>1000', '&' ,'$\sim$',   '\\ \hline'];
 end
latex=[latex;input];

latex=[latex;'\end{tabular}';'}';'\end{adjustbox}';'\end{table}'];
disp(char(latex))









