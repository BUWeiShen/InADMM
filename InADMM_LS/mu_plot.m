%% DATA
randn('seed', 0);
rand('seed',0);
n = 10000; % number of examples
p = 15000; % number of features
sd = 100/p; % sparsity density
x0 = sprandn(p,1,sd);
Q=sprandn(n,p,0.2);
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





param.obj=min(objCHOL,objLSQR);

 withobj=1;
 param.withobj=withobj;
 lsSolver= 'InADMM_prop'; 
 param.lsSolver=lsSolver;





 x=linspace(0,2*mu_Y,100);
 objprop_list=zeros(100,1);
 length_prop_list=zeros(100,1);
 sumiter_prop_list=zeros(100,1);
 mean_prop_list=zeros(100,1);
 max_prop_list=zeros(100,1);
 time_prop_list=zeros(100,1);
 for i=1:100
     mu=x(i);
     sigma_prop = 2/(2+mu*tao);
     param.mu=mu;
     param.sigma_prop=sigma_prop;
    [alpha_prop,history_prop] = InADMM_LeastSquare(param);
     objprop=history_prop.objval(end); % objective value
     length_prop=length(history_prop.cg_iters);% outer iteration number
     sumiter_prop=sum(history_prop.cg_iters);% sum of inner iteration number
     mean_prop= mean(history_prop.cg_iters);% meanvalue of inner iteration numbers
     max_prop=max(history_prop.cg_iters);% maximum of inner iteration numbers
     time_prop=max(history_prop.time);% 
 objprop_list(i,1)=objprop;
 length_prop_list(i,1)=length_prop;
 sumiter_prop_list(i,1)=sumiter_prop;
 mean_prop_list(i,1)=mean_prop;
 max_prop_list(i,1)=max_prop;
 time_prop_list(i,1)=time_prop;

 end



 %% DATA 
n = 10000; % number of examples
p = 15000; % number of features
sd = 100/p; % sparsity density
x0 = sprandn(p,1,sd);
Q=sprandn(n,p,0.5);
q= Q*x0 + 0.1*randn(n,1);

%% set para
l1_ratio=0.5; %lasso


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

mu_Y2=sqrt(2*rho)*normQ;
sigma_2018=2/(2+mu_Y2*tao);







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





param.obj=min(objCHOL,objLSQR);

withobj=1;
 param.withobj=withobj;
 lsSolver= 'InADMM_prop'; 
 param.lsSolver=lsSolver;

 x2=linspace(0,2*mu_Y2,100);
 objprop_list2=zeros(100,1);
 length_prop_list2=zeros(100,1);
 sumiter_prop_list2=zeros(100,1);
 mean_prop_list2=zeros(100,1);
 max_prop_list2=zeros(100,1);
 time_prop_list2=zeros(100,1);
 for i=1:100
     mu=x2(i);
     sigma_prop = 2/(2+mu*tao);
     param.mu=mu;
     param.sigma_prop=sigma_prop;
    [alpha_prop,history_prop] = InADMM_LeastSquare(param);
     objprop=history_prop.objval(end); % objective value
     length_prop=length(history_prop.cg_iters);% outer iteration number
     sumiter_prop=sum(history_prop.cg_iters);% sum of inner iteration number
     mean_prop= mean(history_prop.cg_iters);% meanvalue of inner iteration numbers
     max_prop=max(history_prop.cg_iters);% maximum of inner iteration numbers
     time_prop=max(history_prop.time);% 
 objprop_list2(i,1)=objprop;
 length_prop_list2(i,1)=length_prop;
 sumiter_prop_list2(i,1)=sumiter_prop;
 mean_prop_list2(i,1)=mean_prop;
 max_prop_list2(i,1)=max_prop;
 time_prop_list2(i,1)=time_prop;

 end

%% graph



g_compare=figure;
subplot(2,2,1)
yyaxis left
plot(x,time_prop_list);
axis([0 4500 0 80])
hold on
plot(x(50),time_prop_list(50),'*')
hold off
yyaxis right
plot(x,sumiter_prop_list,'--s');
axis([0 4500 30 120])
hold on
plot(x(50),sumiter_prop_list(50),'bo')
hold off
yyaxis left
xlabel('$\mu$ from 0 to $2\sqrt{2\rho}\|X\|$','Interpreter','latex')
ylabel('Running Time')
yyaxis right
ylabel('Sum of CG')
legend('Running Time','$\mu=\sqrt{2\rho}\|X\|$','Sum of CG','$\mu=\sqrt{2\rho}\|X\|$','Interpreter','latex')
title('(10^4,1.5*10^4,20%)')

subplot(2,2,2)
yyaxis left
plot(x,max_prop_list);
axis([0 4500 0 5])
hold on
plot(x(50),max_prop_list(50),'*')
hold off
yyaxis right
plot(x,mean_prop_list,'--s');
axis([0 4500 0 10])
hold on
plot(x(50),mean_prop_list(50),'bo')
hold off
yyaxis left
xlabel('$\mu$ from 0 to $2\sqrt{2\rho}\|X\|$','Interpreter','latex')
ylabel('Maximum of CG')
yyaxis right
ylabel('Mean of CG')
legend('Maximum of CG','$\mu=\sqrt{2\rho}\|X\|$','Mean of CG','$\mu=\sqrt{2\rho}\|X\|$','Interpreter','latex')
title('(10^4,1.5*10^4,20%)')

subplot(2,2,3)
yyaxis left
t1=plot(x2,time_prop_list2);
axis([0 7500 0 110])
hold on
plot(x2(50),time_prop_list2(50),'*')
hold off
yyaxis right
t2=plot(x2,sumiter_prop_list2,'--s');
axis([0 7500 0 110])
hold on
plot(x2(50),sumiter_prop_list2(50),'bo')
hold off
yyaxis left
xlabel('$\mu$ from 0 to $2\sqrt{2\rho}\|X\|$','Interpreter','latex')
ylabel('Running Time')
yyaxis right
ylabel('Sum of CG')
legend('Running Time','$\mu=\sqrt{2\rho}\|X\|$','Sum of CG','$\mu=\sqrt{2\rho}\|X\|$','Interpreter','latex')
title('(10^4,1.5*10^4,50%)')

subplot(2,2,4)
yyaxis left
plot(x2,max_prop_list2);
axis([0 7500 0 5])
hold on
plot(x2(50),max_prop_list2(50),'*')
hold off
yyaxis right
plot(x2,mean_prop_list2,'--s');
axis([0 7500 0 10])
hold on
plot(x2(50),mean_prop_list2(50),'bo')
hold off
yyaxis left
xlabel('$\mu$ from 0 to $2\sqrt{2\rho}\|X\|$','Interpreter','latex')
ylabel('Maximum of CG')
yyaxis right
ylabel('Mean of CG')
legend('Maximum of CG','$\mu=\sqrt{2\rho}\|X\|$','Mean of CG','$\mu=\sqrt{2\rho}\|X\|$','Interpreter','latex')
title('(10^4,1.5*10^4,50%)')


