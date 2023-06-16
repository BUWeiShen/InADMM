l1_ratio=0.5;
p_list=[100000,200000,300000,400000];
n = 10000;

for order=1:4
        p = p_list(order);
        
        rand('seed', 0);
        randn('seed', 0);
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
        [~, history_InADMM_gdB] = InADMM_EN_Logistic(param);
        
        namestr1=['history_InADMM_gdB' num2str(p) '=history_InADMM_gdB'];
        eval(namestr1);
        
        
        obj_GD=history_InADMM_gdB.objval(end);
        param.obj=obj_GD; 
        withobj=1;
        param.withobj=withobj;
        inSolver='InADMM';
        param.inSolver=inSolver;
        

        [~, history_InADMM_InADMM] = InADMM_EN_Logistic(param);
        
        
        
           namestr11=['history_InADMM_InADMM' num2str(p) '=history_InADMM_InADMM'];
           eval(namestr11);

   



  

end






g_compare_GD_ADMM=figure;
subplot(2,2,1)
t=plot(history_InADMM_gdB100000.time, log(history_InADMM_gdB100000.objval), 'c--s', history_InADMM_InADMM100000.time, log(history_InADMM_InADMM100000.objval), 'b-','MarkerSize', 5,'LineWidth', 1);
set(gca,'Yscale','log')   
ylabel('Objective value');
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('inner=GD','inner=InADMM');
xlabel('Time')
title('p=100000')

subplot(2,2,2)
t=plot(history_InADMM_gdB200000.time, log(history_InADMM_gdB200000.objval), 'c--s', history_InADMM_InADMM200000.time, log(history_InADMM_InADMM200000.objval), 'b-','MarkerSize', 5,'LineWidth', 1);
set(gca,'Yscale','log')   
ylabel('Objective value');
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('inner=GD','inner=InADMM');
xlabel('Time')
title('p=200000')

subplot(2,2,3)
t=plot(history_InADMM_gdB300000.time, log(history_InADMM_gdB300000.objval), 'c--s', history_InADMM_InADMM300000.time, log(history_InADMM_InADMM300000.objval), 'b-','MarkerSize', 5,'LineWidth', 1);
set(gca,'Yscale','log')   
ylabel('Objective value');
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('inner=GD','inner=InADMM');
xlabel('Time')
title('p=300000')

subplot(2,2,4)
t=plot(history_InADMM_gdB400000.time, log(history_InADMM_gdB400000.objval), 'c--s', history_InADMM_InADMM400000.time, log(history_InADMM_InADMM400000.objval), 'b-','MarkerSize', 5,'LineWidth', 1);
set(gca,'Yscale','log')   
ylabel('Objective value');
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('inner=GD','inner=InADMM');
xlabel('Time')
title('p=400000')



