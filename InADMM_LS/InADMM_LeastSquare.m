function [alpha,history] = InADMM_LeastSquare(param)


tStarFunc = tic;


lsSolver=param.lsSolver;
Q=param.Q;
q=param.q;
QTq=param.QTq;
lambda1=param.lambda1;
lambda2=param.lambda2;
rho=param.rho;

withobj=param.withobj;
MAX_ITER = param.MAX_ITER;
ABSTOL   = param.ABSTOL;
RELTOL   = param.RELTOL;

eta=param.eta0;
alpha= param.alpha0;
gamma = param.gamma0;
max_time=param.max_time;

if withobj==1
    obj=param.obj;  
else
    obj=0;
end

QUIET    = 0;


[n,p]=size(Q);



if strcmp(lsSolver, 'InADMM_prop')
       fprintf(' using %s ls solver...\n', lsSolver);
       sigma_prop=param.sigma_prop; mu=param.mu;
elseif strcmp(lsSolver, 'InADMM2018')       
     fprintf(' using %s ls solver...\n', lsSolver);   
        sigma_2018=param.sigma_2018;

elseif strcmp(lsSolver, 'ADMM_fix_tol')
        tol_fix=param.tol_fix;
        fprintf(' using %s ls solver...\n', lsSolver);

elseif strcmp(lsSolver, 'LSQR')
       
        fprintf(' using %s ls solver...\n', lsSolver);
        beta=param.beta0;
        cons1=sqrt(lambda2+rho);
        cons2=1/sqrt(lambda2+rho);
        



elseif strcmp(lsSolver, 'cholesky')
        tStartChol = tic;
        fprintf(' using %s ls solver...\n', lsSolver);
        beta=param.beta0;
        % cache the factorization
        [L, U] = factor(Q, rho);
        tElapsedChol = toc(tStartChol)

end


if ~QUIET
    toc(tStarFunc);
end
%if ~QUIET
    %fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    %'cg iters', 'r norm', 'eps pri', 's norm', 'eps dual','tol', 'objective');
%end
if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n',... 
        'iter','cg iters', 'r norm', 'eps pri', 's norm', 'eps dual','tol','time', 'objective');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADMM LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter=1;
while iter<=MAX_ITER
%% solve LS



        
        
        if strcmp(lsSolver, 'InADMM_prop')
           
            if     iter==1
        tol_part1=0.999;
        tol_part2=0.999;
                    phi_temp=QTq+rho*(alpha+gamma/rho); phi=Q*phi_temp;
                    tol=0.999;
                    [eta,~,~,inner_iter_num,~] = cgs_at_least_1(@afun,phi,tol,1,[],[],eta);
                    QTE=Q'*eta; beta=(phi_temp-QTE)/(lambda2+rho);
                   
            else
        tol_part1=sigma_prop*norm(afun(eta)-phi,2);
        tol_part2=(mu/sqrt(2))*(    sqrt(s_norm_temp^2/rho+rho*r_norm_temp^2)     );
                    tolAbs=0.99*sigma_prop*(norm(afun(eta)-phi,2)+(mu/sqrt(2))*(    sqrt(s_norm_temp^2/rho+rho*r_norm_temp^2)     ) );
                    phi_temp=QTq+rho*(alpha+gamma/rho); phi=Q*phi_temp; tol=tolAbs /norm(phi,2);
                    [eta,~,~,inner_iter_num,~] = cgs(@afun,phi,tol,200,[],[],eta); 
                    QTE=Q'*eta; beta=(phi_temp-QTE)/(lambda2+rho);
        
            end


            


        elseif strcmp(lsSolver, 'InADMM2018')
           
                     phi_temp=QTq+rho*(alpha+gamma/rho); phi=Q*phi_temp;     
                     relres_old=norm(afun(eta)-phi,2)/norm(phi,2); tol=0.99*sigma_2018*relres_old;
                     [eta,~,~,inner_iter_num,~] = cgs(@afun,phi,tol,200,[],[],eta);
                     QTE=Q'*eta; beta=(phi_temp-QTE)/(lambda2+rho);

        elseif strcmp(lsSolver, 'ADMM_fix_tol')
                
                    phi_temp=QTq+rho*(alpha+gamma/rho); phi=Q*phi_temp;  tol=tol_fix;
                    [eta,~,~,inner_iter_num,~] = cgs(@afun,phi,tol,5000,[],[],eta);
                    QTE=Q'*eta; beta=(phi_temp-QTE)/(lambda2+rho);
        
        elseif strcmp(lsSolver, 'cholesky')
                                   
                    qq = QTq + (rho*alpha +gamma);    % temporary value
                    if( n >= p )    % if skinny
                       beta = U \ (L \ qq);
                    else            % if fat
                       eta_temp=U \ ( L \ (Q*qq) );
                       beta = (qq-Q'*eta_temp)/(lambda2+rho);
                    end       
                    inner_iter_num=0; tol=0; 

        elseif strcmp(lsSolver, 'LSQR')
                           
                    [beta, ~, ~, inner_iter_num] = lsqr([Q; cons1*speye(p)], [q; cons2*(rho*alpha+gamma)], [], 50, [], [], beta);  
                     tol=0; 

        end
        

    %% alpha-update 
    alphaold = alpha;
    alpha = shrinkage(beta - gamma/rho,lambda1/rho);

    %% gamma-update
    gamma = gamma -rho*(beta-alpha);
    %% norms
    s_norm_temp  = norm(rho*(alpha - alphaold));
    r_norm_temp = norm(beta -alpha);

    %% diagnostics, reporting, termination checks
    k=iter;
    history.objval(k)  = objective(alpha);  
    history.cg_iters(k) = inner_iter_num;
    history.r_norm(k)   = r_norm_temp;
    history.s_norm(k)  = s_norm_temp;       
    history.tol(k) = tol;
    history.eps_pri(k) = sqrt(p)*ABSTOL + RELTOL*max(norm(beta), norm(-alpha));
    history.eps_dual(k)= sqrt(p)*ABSTOL + RELTOL*norm(gamma);

    if strcmp(lsSolver, 'InADMM_prop')
    history.tol_part1(k) = tol_part1;
    history.tol_part2(k) = tol_part2;
    end
    history.time(k)=toc(tStarFunc);

    
 %{
        if ~QUIET
        fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.8f\n', k, ...
              sum(history.cg_iters), history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k),history.tol(k), history.time(k));
        end 
 %}

            if ~QUIET
        fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.8f\t%10.8f\t%10.8f\n', ...
            k, sum(history.cg_iters), history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k),history.tol(k), history.time(k),history.objval(k));
            end 

            

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
    iter=iter+1;
end

if ~QUIET
    toc(tStarFunc);
end

function obj = objective(alpha)
 
    obj = 0.5*norm(Q*alpha-q,2)^2+ lambda1*norm(alpha,1)+0.5*lambda2*norm(alpha,2)^2;
end


function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end

 function y = afun(eta)
 
  
  temp=Q'*eta;
  y=Q*temp+(rho+lambda2)*eta;
 end


function [L, U] = factor(Q, rho)
   
  

    if ( n >= p )    % if skinny
       L = chol( Q'*Q + (lambda2+rho)*speye(p), 'lower' );
    else            % if fat
       L = chol( Q*Q'+ (lambda2+rho)*speye(n), 'lower' );
    end
    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end






end








