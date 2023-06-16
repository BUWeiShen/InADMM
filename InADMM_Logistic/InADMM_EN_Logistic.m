function [alpha, history] = InADMM_EN_Logistic(param)


% InADMM for elatic-net logistic regression using another InADMM for
% beta-subproblem

% inputs: 
%   parameter structure containing input variables 

% output: 
%   alpha   - alpha at final iteration
%   his      - history matrix. The columns correspond to: 
%              1) iteration number 
%              2) training misfit 
%              3) validation misfit
%              4) training accuracies, 
%              5) validation accuracies, 
%              6) number of newton iterations for z-step 
%              7) z-step misfit 
%              8) relative gradient norm, 
%              9)  
%              10) primal residual, 
%              11) primal tolerance, 
%              12) dual residual, 
%              13) dual tolerance, 
%              14) 
%              15) current runtime of algorithm
tStarFunc = tic;


inSolver=param.inSolver;


% parameters for input data
A          = param.A;
b          = param.b;

% parameters for elastice logistic model
C          = param.C;
lambda_1   = param.lambda_1;
lambda_2   = param.lambda_2;

% parameters for beta-step solver
rho_outer  = param.rho_outer;
rho_inner  = param.rho_inner;
mu_outer   = param.mu_outer;
mu_inner   = param.mu_inner;
nu         = param.nu;


withobj=param.withobj;
MAX_ITER = param.MAX_ITER;
ABSTOL   = param.ABSTOL;
RELTOL   = param.RELTOL;


max_time=param.max_time;

if withobj==1
    obj=param.obj;  
else
    obj=0;
end
%% Global constants and defaults
QUIET    = 0;




      
   
%% Data preprocessing
[n, p] = size(A);
tilde_A=[-b,-A];




if strcmp(inSolver, 'InADMM')
       fprintf(' using %s inner solver...\n', inSolver);
       alpha = zeros(p+1,1); gamma = zeros(p+1,1); eta_update=zeros(n,1); alpha_update=zeros(n,1); 
       gamma_update=zeros(n,1); alphaold_update=zeros(n,1); gammaold_update=zeros(n,1);  

elseif  strcmp(inSolver, 'InADMM_fix_tol') 
        tol_fix=param.tol_fix;
        fprintf(' using %s inner solver...\n', inSolver);
        beta = zeros(p+1,1); alpha = zeros(p+1,1);
        gamma = zeros(p+1,1); eta_update=zeros(n,1); alpha_update=zeros(n,1);
        gamma_update=zeros(n,1); alphaold_update=zeros(n,1); gammaold_update=zeros(n,1);  
elseif strcmp(inSolver, 'GD')       
       fprintf(' using %s inner solver...\n', inSolver);   
        alpha = zeros(p+1,1); gamma = zeros(p+1,1);

elseif  strcmp(inSolver, 'GD_fix_tol') 
        tol_fix=param.tol_fix;
        fprintf(' using %s inner solver...\n', inSolver);
        beta = zeros(p+1,1); alpha = zeros(p+1,1); gamma = zeros(p+1,1); 
end


if ~QUIET
    toc(tStarFunc);
end

if ~QUIET
    fprintf('%3s\t%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter','sum_inner_iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective1','objective');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADMM LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter=1;
while iter<=MAX_ITER


         if strcmp(inSolver, 'InADMM')
           
            if     iter==1
                % beta-update
                beta = zeros(p+1,1);
                inner_iter_num=1;
                % gradient norm
                EE1=min(exp(tilde_A*beta)./(1 + exp(tilde_A*beta)),1);
                g = C/n*tilde_A'*EE1+lambda_2*beta+rho_outer*(beta- alpha + gamma);
                g_old_norm=norm(g,2);
                   
            else
                x0=beta; 
                eta0=eta_update;
                alpha0=alpha_update;
                gamma0=gamma_update;
                alphaold=alphaold_update;
                gammaold=gammaold_update;   
                xi_k=alpha-gamma;
                sigma=2/(2+mu_outer/sqrt(lambda_2));
                g_tol=0.99*sigma*(g_old_norm+(mu_outer/sqrt(2))*(    sqrt(s_norm_temp^2/rho_outer+rho_outer*r_norm_temp^2)     ) );
                [inner_iter_num,g_old_norm,beta,eta_update,alpha_update,gamma_update,alphaold_update,gammaold_update]  = update_x(x0,eta0,alpha0,gamma0,alphaold,gammaold,xi_k,g_tol);

            end


            


        elseif strcmp(inSolver, 'GD')
           
                    if     iter==1
                            beta = zeros(p+1,1);
                            inner_iter_num=1;
                            % gradient norm
                            EE1=min(exp(tilde_A*beta)./(1 + exp(tilde_A*beta)),1);
                            g1 = C/n*tilde_A'*EE1+lambda_2*beta+rho_outer*(beta- alpha + gamma);
                            g_old_norm=norm(g1,2);
                    else
                            sigma=2/(2+mu_outer/sqrt(lambda_2));
                            g_tol=0.99*sigma*(g_old_norm+(mu_outer/sqrt(2))*(    sqrt(s_norm_temp^2/rho_outer+rho_outer*r_norm_temp^2)     ) );
                            [beta,inner_iter_num,g_old_norm] = update_x_GD( gamma,alpha, beta,g_tol);
                           
                    end

        elseif strcmp(inSolver, 'InADMM_fix_tol')
                
                   
                x0=beta; 
                eta0=eta_update;
                alpha0=alpha_update;
                gamma0=gamma_update;
                alphaold=alphaold_update;
                gammaold=gammaold_update;   
                xi_k=alpha-gamma;
                
                g_tol=tol_fix;
                [inner_iter_num,g_old_norm,beta,eta_update,alpha_update,gamma_update,alphaold_update,gammaold_update]  = update_x(x0,eta0,alpha0,gamma0,alphaold,gammaold,xi_k,g_tol);

        
        elseif strcmp(inSolver, 'GD_fix_tol')
                                   
                  g_tol=tol_fix;
                 [beta,inner_iter_num,~] = update_x_GD( gamma,alpha, beta,g_tol);

        end

    % alpha-update with relaxation
    alphaold = alpha;
    alpha = beta + gamma;
    alpha(2:end) = shrinkage(alpha(2:end), lambda_1/rho_outer);
   
    % gamma-update      
    gamma = gamma + (beta - alpha);
    
 
  
    % diagnostics, reporting, termination checks
    s_norm_temp  = norm(rho_outer*(alpha - alphaold));
    r_norm_temp = norm(beta- alpha);

      

    k=iter;
    history.iters(k)  = inner_iter_num;
    history.objval1(k)  = objective1( beta, alpha);
    history.objval(k)  = objective2( alpha);
    history.r_norm(k)  = r_norm_temp;
    history.s_norm(k)  = s_norm_temp;
    history.eps_pri(k) = sqrt(p)*ABSTOL + RELTOL*max(norm(beta), norm(alpha));
    history.eps_dual(k)= sqrt(p)*ABSTOL + RELTOL*norm(rho_outer*gamma);
    history.time(k)=toc(tStarFunc);

   if ~QUIET
        fprintf('%3d\t%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', k,sum(history.iters), ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k),history.eps_dual(k), history.objval1(k), history.objval(k));
   end

if ~(iter==1)
                if (history.r_norm(k) < history.eps_pri(k) && ...
                      history.s_norm(k) < history.eps_dual(k)&&(~withobj))
                       break;
                end
                   
                if(history.objval(k)<=obj&&withobj)
                         break;
                end
                
                if history.time(k)>max_time
                    disp(char(['time out at ', num2str(history.time(k))]))
                    break;
                end
end


iter=iter+1;
end

if ~QUIET
    toc(tStarFunc);
end











function obj = objective1(x, z)   
    obj = (C/n)*sum(log(1 + exp(tilde_A*x))) + lambda_2*0.5*norm(x)^2+lambda_1*norm(z(2:end),1);
end
function obj = objective2(z)
% force to meet the constrained condition x=z     
    obj = (C/n)*sum(log(1 + exp(tilde_A*z))) + lambda_2*0.5*norm(z)^2+lambda_1*norm(z(2:end),1);
end


function z = shrinkage(a, kappa)
    z = max(0, a-kappa) - max(0, -a-kappa);
end



    
    
function [T,g_old_norm,x,eta,z,u,zold,uold] = update_x(x0,eta0,alpha0,gamma0,alphaold,gammaold,xi_k,g_tol)

% Solve L2 regularized logistic regression via InADMM
% minimize  C/n sum( log(1 + exp(-b_i*(a_i'w + v)) ) + lambda_2/2*norm(x,2)^2 + rho_outer/2||x-xi_k||_2^2
% 


%% Global constants and defaults
MAX_ITER_inner = 5000;



%% ADMM solver
eta=eta0;
z = alpha0;
u = gamma0;
zold=alphaold;
uold=gammaold;



%k=1
    % x-update
    
for T = 1:MAX_ITER_inner



if T==1
    x=x0;
    phi_temp=(rho_outer/rho_inner)*xi_k+tilde_A'*(zold-uold);
    phi=tilde_A*phi_temp;      
else
    tau_inner=sqrt(nu)*rho_inner^2/(lambda_2+rho_outer)^(1.5);
    sigma_inner=2/(2+mu_inner*tau_inner);
    % x-update   
    abs_tol=0.99*sigma_inner*(norm(afun(eta)-phi,2)+(mu_inner/sqrt(2))*(    sqrt(s_norm_temp_inner^2/rho_inner+rho_inner*r_norm_temp_inner^2)     ) );
    
    phi_temp=(rho_outer/rho_inner)*xi_k+tilde_A'*(z-u);
    phi=tilde_A*phi_temp;
       
    tol=abs_tol/norm(phi,2);

    [eta,~,~,~,~] = cgs(@afun,phi,tol,200,[],[],eta);
    x=(rho_inner/(lambda_2+rho_outer))*(phi_temp-tilde_A'*eta);
end    
    % z-update with relaxation
    zold = z; 
    z=update_alpha(x,u,zold);

    % u-update
    uold=u;
    u = u + (tilde_A*x - z);
    
    % norms
    s_norm_temp_inner  = norm(rho_inner*(z - zold));
    r_norm_temp_inner = norm(u-uold);
    



    % gradient
    EE1_inner=min(exp(tilde_A*x)./(1 + exp(tilde_A*x)),1);
    g_inner = C/n*tilde_A'*EE1_inner+lambda_2*x+rho_outer*(x-xi_k);
    g_old_norm=norm(g_inner,2);

    if g_old_norm<g_tol
        break;
    end
end

end

    function alpha=update_alpha(beta_innner,gamma_inner,alpha_inner0)

     %   minimize [ -c/n*logistic(alpha_i) + (rho_inner/2)(alpha_i - qq^k_i)^2 ]
     % via Newton's method; for a single subsystem only. 
     
    qq=tilde_A*beta_innner+gamma_inner;
    ALPHA = 0.1;
    BETA  = 0.5;
    TOLERANCE = 1e-20;
    MAX_ITER_alpha = 50;
    alpha=zeros(n,1);
    
    for i=1:n   
    al_i=alpha_inner0(i);
    f = @(w) (C/n*log(1 + exp(w)) + (rho_inner/2)*(w - qq(i))^2);
    for iter_alpha = 1:MAX_ITER_alpha
        fx = f(al_i);
        ED1PE=min(1,exp(al_i)/(1 + exp(al_i))); 
        ED1PE2=max(0,exp(al_i)/(1 + exp(al_i))^2);
        al_g = C/n*ED1PE+ rho_inner*(al_i - qq(i));
        H = C/n*ED1PE2 + rho_inner;
        dx = -H\al_g;   % Newton step
        dfx = al_g'*dx; % Newton decrement
        if abs(dfx) < TOLERANCE
            break;
        end
        % backtracking
        t = 1;
        while f(al_i + t*dx) > fx + ALPHA*t*dfx
            t = BETA*t;
        end
        al_i = al_i + t*dx;
    end
    alpha(i)=al_i;
    end   
 end
 
  function y = afun(eta)

  temp=tilde_A'*eta;
  y=tilde_A*temp+((lambda_2+rho_outer)/rho_inner).*eta;
  end



function [x,iter,g_norm] = update_x_GD( u, z, x0,g_tol)
    % solve the x update
    %   minimize [ -logistic(x_i) +lambda_2*0.5*norm(x_i)^2 + (rho/2)||x_i - z^k + u^k||^2 ]
    % via Newton's method; for a single subsystem only.
    ALPHA = 0.1;
    BETA  = 0.5;
    
    MAX_ITER_GD = 5000;
   
   
    if exist('x0', 'var')
        x = x0;
    else
        x = zeros(p+1,1);
    end
    
    tilde_A = [-b,-A];
    f = @(w) ((C/n)*sum(log(1 + exp(tilde_A*w)))  + lambda_2*0.5*norm(w)^2 + (rho_outer/2)*norm(w - z + u).^2);
        EE=min(exp(tilde_A*x)./(1 + exp(tilde_A*x)),1);
        g = (C/n)*tilde_A'*EE + (lambda_2+rho_outer)*x+ rho_outer*(- z + u);
        dx = -g;   % descent direction
        dfx = g'*dx; %  decrement
        
    for iter = 1:MAX_ITER_GD
        fx = f(x);
        
        %% backtracking
        t = 1;
        while f(x + t*dx) > fx + ALPHA*t*dfx
            t = BETA*t;
        end
        x = x + t*dx;  
        EE=min(exp(tilde_A*x)./(1 + exp(tilde_A*x)),1);
        g = (C/n)*tilde_A'*EE + (lambda_2+rho_outer)*x+ rho_outer*(- z + u);
        dx = -g;   % descent direction
        dfx = g'*dx; %  decrement
        g_norm=norm(g);
       
        if g_norm<g_tol
            break;
        end        
    end    
end



end



 