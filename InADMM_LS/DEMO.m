%% table header
caption='\caption{LASSO ($a=1$): Comparison between $\text{InADMM}_{Prop}$, InADMM(2018), $\text{ADMM}_{Cholesky}$, $\text{ADMM}_{LSQR}$ and $\text{ADMM}_{1e-t}$}';
label='\label{tab:elasticNet5}';
begin_adjustment='\begin{adjustbox}{center,max width=1\textwidth}';
scale_box='\scalebox{0.85}{';
begin_tabular='\begin{tabular}{|c|l|l|l|l|l|l|}';
headerline='$(n,p,sd) $  & Algorithm & Iteration & Sum CG & Mean/Max CG & Time & Obj \\ \hline';
columntitle1='\multirow{9}{*}{$(2.5*10^4,4*10^4)$}';
latex22 = {'\begin{table}[]';caption;label;begin_adjustment;scale_box;begin_tabular;'\hline';headerline;columntitle1};


randn('seed', 0);
rand('seed',0);
n = 25000; % number of examples
p = 40000; % number of features
sd = 100/p; % sparsity density
x0 = sprandn(p,1,sd);
Q=randn(n,p);
q= Q*x0 + 0.1*randn(n,1);

%% set para
l1_ratio=1; %lasso


r_max = norm(Q'*q,'inf');
rho=0.05*r_max;
lambda= 0.1*r_max;

lambda1=l1_ratio*lambda;
lambda2=(1-l1_ratio)*lambda;


max_time=1000;
MAX_ITER = 500;
ABSTOL   = 1e-4;
RELTOL   = 1e-3;

eta0=zeros(n,1);
beta0=zeros(p,1);
alpha0= zeros(p,1);
gamma0 = zeros(p,1);

QTq=Q'*q;


normQ=normest(Q,0.0001);
tao=1/(lambda2+rho);

mu_Y=sqrt(2*rho)*normQ;
sigma_2018=2/(2+mu_Y*tao);







param.Q=Q;
param.q=q;
param.QTq=QTq;
param.lambda1=lambda1;
param.lambda2=lambda2;
param.rho=rho;

param.sigma_2018=sigma_2018;

param.MAX_ITER = MAX_ITER;
param.ABSTOL   = ABSTOL;
param.RELTOL   = RELTOL;
param.eta0=eta0;
param.alpha0= alpha0;
param.gamma0 = gamma0;
param.beta0=beta0;

param.max_time=max_time;

% for Xie et al 2017
param.sigma_Xie=0.99;
param.w0=beta0;
param.A= speye(p);
param.B=-speye(p);
param.b=0;

%% LSQR
lsSolver= 'LSQR'; 
withobj=0;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_LSQR,history_LSQR] = InADMM_LeastSquare(param);

 objLSQR=history_LSQR.objval(end); % objective value
 length_LSQR=length(history_LSQR.cg_iters);% outer iteration number
 sumiter_LSQR=sum(history_LSQR.cg_iters);% sum of inner iteration number
 mean_LSQR= mean(history_LSQR.cg_iters);% meanvalue of inner iteration numbers
 max_LSQR=max(history_LSQR.cg_iters);% maximum of inner iteration numbers
 time_LSQR=max(history_LSQR.time);% 
 if time_LSQR<1000
    input=['&$\text{ADMM}_{LSQR}$  &', num2str(length_LSQR), '&', '$\sim$', '&', '$\sim$',  '&' ,num2str(time_LSQR), '&' ,num2str(objLSQR),   '\\ '];
 
 else
    input=['&$\text{ADMM}_{LSQR}$   &','$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ '];
 end

latex22=[latex22;input];

%% Cholesky
lsSolver= 'cholesky'; 
withobj=0;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_CHOL,history_CHOL] = InADMM_LeastSquare(param);

 objCHOL=history_CHOL.objval(end); % objective value
 length_CHOL=length(history_CHOL.cg_iters);% outer iteration number
 sumiter_CHOL=sum(history_CHOL.cg_iters);% sum of inner iteration number
 mean_CHOL= mean(history_CHOL.cg_iters);% meanvalue of inner iteration numbers
 max_CHOL=max(history_CHOL.cg_iters);% maximum of inner iteration numbers
 time_CHOL=max(history_CHOL.time);% 
 if time_CHOL<1000
input=['&$\text{ADMM}_{Cholesky}$  &', num2str(length_CHOL), '&', '$\sim$', '&', '$\sim$',  '&' ,num2str(time_CHOL), '&' ,num2str(objCHOL),   '\\ '];
 else
input=['&$\text{ADMM}_{Cholesky}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ '];
 end
latex22=[latex22;input];



%% InADMM2018
lsSolver= 'InADMM2018'; % 'ADMM_fix_tol','LSQR','cholesky'
if time_CHOL<1000 || time_LSQR<1000
withobj=1;
param.obj=min(objCHOL,objLSQR);
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_2018,history_2018] = InADMM_LeastSquare(param);
 obj2018=history_2018.objval(end); % objective value
 length_2018=length(history_2018.cg_iters);% outer iteration numbernargin
 sumiter_2018=sum(history_2018.cg_iters);% sum of inner iteration number
 mean_2018= mean(history_2018.cg_iters);% meanvalue of inner iteration numbers
 max_2018=max(history_2018.cg_iters);% maximum of inner iteration numbers
 time_2018=max(history_2018.time);% 
input=['&InADMM(2018)  &', num2str(length_2018), '&', num2str(sumiter_2018), '&', num2str(mean_2018), '/' ,num2str(max_2018),  '&' ,num2str(time_2018), '&' ,num2str(obj2018),   '\\ '];
latex22=[latex22;input];
else
  withobj=0;

param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_2018,history_2018] = InADMM_LeastSquare(param);
 obj2018=history_2018.objval(end); % objective value
 param.obj=obj2018;
 length_2018=length(history_2018.cg_iters);% outer iteration number
 sumiter_2018=sum(history_2018.cg_iters);% sum of inner iteration number
 mean_2018= mean(history_2018.cg_iters);% meanvalue of inner iteration numbers
 max_2018=max(history_2018.cg_iters);% maximum of inner iteration numbers
 time_2018=max(history_2018.time);% 
input=['&InADMM(2018)  &', num2str(length_2018), '&', num2str(sumiter_2018), '&', num2str(mean_2018), '/' ,num2str(max_2018),  '&' ,num2str(time_2018), '&' ,num2str(obj2018),   '\\ '];
latex22=[latex22;input];  
end

%% Xie
lsSolver= 'Xie_et_al_2017_eta'; 
withobj=1;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_xie,history_xie] = InADMM_LeastSquare_Xie(param);
 objxie=history_xie.objval(end); % objective value
 length_xie=length(history_xie.cg_iters);% outer iteration number
 sumiter_xie=sum(history_xie.cg_iters);% sum of inner iteration number
 mean_xie= mean(history_xie.cg_iters);% meanvalue of inner iteration numbers
 max_xie=max(history_xie.cg_iters);% maximum of inner iteration numbers
 time_xie=max(history_xie.time);% 
 if time_xie<1000
    input=['& ADMM (Xie et al 2017)  &', num2str(length_xie), '&', num2str(sumiter_xie), '&', num2str(mean_xie), '/' ,num2str(max_xie),  '&' ,num2str(time_xie), '&' ,num2str(objxie),   '\\ '];
 
 else
    input=['&ADMM (Xie et al 2017) &','$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ '];
 end

latex22=[latex22;input];

%% Proposed InADMM 1
lsSolver= 'InADMM_prop'; %'InADMM2018', 'ADMM_fix_tol','LSQR','cholesky'
withobj=1;
mu=1*mu_Y;
sigma_prop = 2/(2+mu*tao);
param.mu=mu;
param.sigma_prop=sigma_prop;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_prop,history_prop] = InADMM_LeastSquare(param);
 objprop=history_prop.objval(end); % objective value
 length_prop=length(history_prop.cg_iters);% outer iteration number
 sumiter_prop=sum(history_prop.cg_iters);% sum of inner iteration number
 mean_prop= mean(history_prop.cg_iters);% meanvalue of inner iteration numbers
 max_prop=max(history_prop.cg_iters);% maximum of inner iteration numbers
 time_prop=max(history_prop.time);% 
 input=['&   $\textbf{InADMM}_{\mu=\sqrt{2\rho}\|X\|}$  &', num2str(length_prop), '&', num2str(sumiter_prop), '&', num2str(mean_prop), '/' ,num2str(max_prop),  '&' ,num2str(time_prop), '&' ,num2str(objprop),   '\\ \cline{2-7}'];
latex22=[latex22;input];




%% fix tol ADMM 
% 1e-2
lsSolver= 'ADMM_fix_tol'; % ,'LSQR','cholesky'
withobj=1;
tol_fix=1e-2;
param.tol_fix=tol_fix;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_fix_tol2,history_fixtol2] = InADMM_LeastSquare(param);

 objfixtol2=history_fixtol2.objval(end); % objective value
 length_fixtol2=length(history_fixtol2.cg_iters);% outer iteration number
 sumiter_fixtol2=sum(history_fixtol2.cg_iters);% sum of inner iteration number
 mean_fixtol2= mean(history_fixtol2.cg_iters);% meanvalue of inner iteration numbers
 max_fixtol2=max(history_fixtol2.cg_iters);% maximum of inner iteration numbers
 time_fixtol2=max(history_fixtol2.time);% 
 if length_fixtol2<500 && time_fixtol2<1000
    input=['&$\text{ADMM}_{1e-2}$  &', num2str(length_fixtol2), '&', num2str(sumiter_fixtol2), '&', num2str(mean_fixtol2), '/' ,num2str(max_fixtol2),  '&' ,num2str(time_fixtol2), '&' ,num2str(objfixtol2),   '\\ '];
 elseif length_fixtol2>=500
    input=['&$\text{ADMM}_{1e-2}$  &', '$>500$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_fixtol2>=1000
     input=['&$\text{ADMM}_{1e-2}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ '];
 end
latex22=[latex22;input];
%% 1e-4
 lsSolver= 'ADMM_fix_tol'; % ,'LSQR','cholesky'
withobj=1;
tol_fix=1e-4;
param.tol_fix=tol_fix;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_fix_tol4,history_fixtol4] = InADMM_LeastSquare(param);


 objfixtol4=history_fixtol4.objval(end); % objective value
 length_fixtol4=length(history_fixtol4.cg_iters);% outer iteration number
 sumiter_fixtol4=sum(history_fixtol4.cg_iters);% sum of inner iteration number
 mean_fixtol4= mean(history_fixtol4.cg_iters);% meanvalue of inner iteration numbers
 max_fixtol4=max(history_fixtol4.cg_iters);% maximum of inner iteration numbers
 time_fixtol4=max(history_fixtol4.time);% 
 if length_fixtol4<500 && time_fixtol4<1000
    input=['&$\text{ADMM}_{1e-4}$  &', num2str(length_fixtol4), '&', num2str(sumiter_fixtol4), '&', num2str(mean_fixtol4), '/' ,num2str(max_fixtol4),  '&' ,num2str(time_fixtol4), '&' ,num2str(objfixtol4),   '\\ '];
 elseif length_fixtol4>=500
    input=['&$\text{ADMM}_{1e-4}$  &', '$>500$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_fixtol4>=1000
     input=['&$\text{ADMM}_{1e-4}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ '];
 end
 latex22=[latex22;input];
%% 1e-6
 lsSolver= 'ADMM_fix_tol'; % ,'LSQR','cholesky'
withobj=1;
tol_fix=1e-6;
param.tol_fix=tol_fix;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_fix_tol6,history_fixtol6] = InADMM_LeastSquare(param);


 objfixtol6=history_fixtol6.objval(end); % objective value
 length_fixtol6=length(history_fixtol6.cg_iters);% outer iteration number
 sumiter_fixtol6=sum(history_fixtol6.cg_iters);% sum of inner iteration number
 mean_fixtol6= mean(history_fixtol6.cg_iters);% meanvalue of inner iteration numbers
 max_fixtol6=max(history_fixtol6.cg_iters);% maximum of inner iteration numbers
 time_fixtol6=max(history_fixtol6.time);% 
 if length_fixtol6<500 && time_fixtol6<1000
    input=['&$\text{ADMM}_{1e-6}$  &', num2str(length_fixtol6), '&', num2str(sumiter_fixtol6), '&', num2str(mean_fixtol6), '/' ,num2str(max_fixtol6),  '&' ,num2str(time_fixtol6), '&' ,num2str(objfixtol6),   '\\ '];
 elseif length_fixtol6>=500
    input=['&$\text{ADMM}_{1e-6}$  &', '$>500$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_fixtol6>=1000
     input=['&$\text{ADMM}_{1e-6}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ '];
 end
latex22=[latex22;input];
%% 1e-8
 lsSolver= 'ADMM_fix_tol'; % ,'LSQR','cholesky'
withobj=1;
tol_fix=1e-8;
param.tol_fix=tol_fix;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_fix_tol8,history_fixtol8] = InADMM_LeastSquare(param);

 objfixtol8=history_fixtol8.objval(end); % objective value
 length_fixtol8=length(history_fixtol8.cg_iters);% outer iteration number
 sumiter_fixtol8=sum(history_fixtol8.cg_iters);% sum of inner iteration number
 mean_fixtol8= mean(history_fixtol8.cg_iters);% meanvalue of inner iteration numbers
 max_fixtol8=max(history_fixtol8.cg_iters);% maximum of inner iteration numbers
 time_fixtol8=max(history_fixtol8.time);% 
 if length_fixtol8<500 && time_fixtol8<1000
    input=['&$\text{ADMM}_{1e-8}$  &', num2str(length_fixtol8), '&', num2str(sumiter_fixtol8), '&', num2str(mean_fixtol8), '/' ,num2str(max_fixtol8),  '&' ,num2str(time_fixtol8), '&' ,num2str(objfixtol8),   '\\ '];
 elseif length_fixtol8>=500
    input=['&$\text{ADMM}_{1e-8}$  &', '$>500$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ '];
 elseif time_fixtol8>=1000
     input=['&$\text{ADMM}_{1e-8}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ '];
 end
latex22=[latex22;input];
%% 1e-10
 lsSolver= 'ADMM_fix_tol'; % ,'LSQR','cholesky'
withobj=1;
tol_fix=1e-10;
param.tol_fix=tol_fix;
param.lsSolver=lsSolver;
param.withobj=withobj;
[alpha_fix_tol10,history_fixtol10] = InADMM_LeastSquare(param);
 objfixtol10=history_fixtol10.objval(end); % objective value
 length_fixtol10=length(history_fixtol10.cg_iters);% outer iteration number
 sumiter_fixtol10=sum(history_fixtol10.cg_iters);% sum of inner iteration number
 mean_fixtol10= mean(history_fixtol10.cg_iters);% meanvalue of inner iteration numbers
 max_fixtol10=max(history_fixtol10.cg_iters);% maximum of inner iteration numbers
 time_fixtol10=max(history_fixtol10.time);% 
 if length_fixtol10<500 && time_fixtol10<1000
    input=['&$\text{ADMM}_{1e-10}$  &', num2str(length_fixtol10), '&', num2str(sumiter_fixtol10), '&', num2str(mean_fixtol10), '/' ,num2str(max_fixtol10),  '&' ,num2str(time_fixtol10), '&' ,num2str(objfixtol10),   '\\ \hline'];
 elseif length_fixtol10>=500
    input=['&$\text{ADMM}_{1e-10}$  &', '$>500$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$\sim$', '&' ,'$\sim$',   '\\ \hline'];
 elseif time_fixtol10>=1000
     input=['&$\text{ADMM}_{1e-10}$  &', '$\sim$', '&', '$\sim$', '&', '$\sim$',  '&' ,'$>1000$', '&' ,'$\sim$',   '\\ \hline'];
 end
latex22=[latex22;input;'\end{tabular}';'}';'\end{adjustbox}';'\end{table}'];
disp(char(latex22))

