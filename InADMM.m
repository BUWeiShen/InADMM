function [wFinal,wOptLoss,wOptAcc,his] = InADMM(param)



tStarFunc = tic;


C=param.C;
fix_pcg_tol=param.fix_pcg_tol;
W = param.W;
U = param.U;
Z = param.Z;
inMu = param.inMu;
A = param.A;
LLtWrefT = param.LLtWref;
sigma = param.sigma;
sigma_another = param.sigma_another;
maxIter = param.maxIter;
stoppingCrit = param.stoppingCrit;

lambda_2 = param.lambda_2;
lsSolver = param.lsSolver;
rho0 = param.rho0;


Dtrain = param.Dtrain;
Dval = param.Dval;
Ctrain = param.Ctrain;
Cval = param.Cval;
L = param.L;

f = param.f; % class obj func with training data
fVal = param.fVal; % class obj func with validation data

% parameters for Z-step solver
atolZ = param.atolZ;
rtolZ = param.rtolZ;
maxIterZ = param.maxIterZ;
linSolMaxIterZ = param.linSolMaxIterZ;
linSolTolZ = param.linSolTolZ;
lsMaxIterZ = param.lsMaxIterZ;
outZ = param.outZ;





lowestMisfit = Inf;
highestAcc = 0;


nc = size(Ctrain,1);



Ntrain = size(Dtrain,2); Nval = size(Dval,2);

rho = rho0;
rhoOld = rho;

his = zeros(maxIter,6);

if strcmp(lsSolver, 'proposed_in_pcg')
       fprintf(' using %s ls solver...\n', lsSolver);
      
 elseif strcmp(lsSolver, 'another_in_pcg')       
     fprintf(' using %s ls solver...\n', lsSolver);     

elseif strcmp(lsSolver, 'qr')
        tStartQR = tic;
        fprintf(' using %s ls solver...\n', lsSolver);
        [Q,R] = qr(A);
        tElapsedQR = toc(tStartQR)

elseif strcmp(lsSolver, 'fix_tol_pcg')

        fprintf(' using %s ls solver...\n', lsSolver);

elseif strcmp(lsSolver, 'cholesky')
        tStartChol = tic;
        fprintf(' using %s ls solver...\n', lsSolver);
        Cho = chol(A);
        tElapsedChol = toc(tStartChol)

end














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADMM LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter=1;

while iter<=maxIter

        
        
        %% solve LS
        
        Wold = W;
        
        
        if strcmp(lsSolver, 'proposed_in_pcg')
           
            if     iter==1
        
                   currentRuntime = toc(tStarFunc)
                    temp_rhs=Dtrain*(Z+U)'; rhs=rho*temp_rhs + lambda_2*LLtWrefT;
                   resM=A*W' - rhs; 
                   
            else
        
                    temp_rhs1=Dtrain*(Z+U)'; rhs=rho*temp_rhs1 + lambda_2*LLtWrefT;
            
                    for inner_i=1:nc
                        
                        w_old=Wold(inner_i,:)';
                        normZ=norm(Zold(inner_i,:)-Z(inner_i,:)); normU=norm(Uold(inner_i,:)-U(inner_i,:));
                        normM=sqrt(rho)*sqrt(normZ^2+normU^2); tolAbs=sigma*(norm(resM(:,inner_i))+(inMu/sqrt(2))*normM);
                        rhs_i=rhs(:,inner_i); tol=tolAbs/norm(rhs_i,2);
                        [w,~,~,~] = pcg(A,rhs_i,tol,500,[],[],w_old); W(inner_i,:)=w';
                        currentRuntime = toc(tStarFunc)
                              if toc(tStarFunc)>=stoppingCrit{2}  
                                  break; 
                              end
                    end
            
            
                    resM=A*W' - rhs; 
        
            end


            


        elseif strcmp(lsSolver, 'another_in_pcg')
           
            if     iter==1
        
                  currentRuntime = toc(tStarFunc)
                    temp_rhs=Dtrain*(Z+U)'; rhs=rho*temp_rhs + lambda_2*LLtWrefT;
                   resM=A*W' - rhs; 
                   
            else
        
                    temp_rhs1=Dtrain*(Z+U)'; rhs=rho*temp_rhs1 + lambda_2*LLtWrefT;
            
                    for inner_i=1:nc
                        
                        w_old=Wold(inner_i,:)'; rhs_i=rhs(:,inner_i); 
                        tolAbs=sigma_another*norm(A*w_old-rhs_i);
                        tol=tolAbs/norm(rhs_i,2);
                        [w,~,~,~] = pcg(A,rhs_i,tol,500,[],[],w_old); W(inner_i,:)=w';
                        currentRuntime = toc(tStarFunc)
                              if toc(tStarFunc)>=stoppingCrit{2}  
                                  break; 
                              end                      
                    end
            
            
                    resM=A*W' - rhs; 
        
            end
        elseif strcmp(lsSolver, 'fix_tol_pcg')
                temp_rhs1=Dtrain*(Z+U)';
                rhs=rho*temp_rhs1 + lambda_2*LLtWrefT;
                for inner_i=1:nc
                    
                    w_old=Wold(inner_i,:)';
                    rhs_i=rhs(:,inner_i);
                    [w,~,~,~] = pcg(A,rhs_i,fix_pcg_tol,500,[],[],w_old); W(inner_i,:)=w';
                    currentRuntime = toc(tStarFunc)
                              if toc(tStarFunc)>=stoppingCrit{2}  
                                  break; 
                              end
                end
                
        
        
        elseif strcmp(lsSolver, 'cholesky')
                
                temp_rhs1=Dtrain*(Z+U)'; rhs=rho*temp_rhs1 + lambda_2*LLtWrefT;
                if rhoOld~=rho
                    A = rho*DDt + lambda_2*LLt;
                    Cho = chol(A);
                end
                W = Cho\(Cho'\rhs);
               
                W = W'; % transpose back
        
        
        elseif strcmp(lsSolver, 'qr')
                % if rho did not change, do not re-factorize
                temp_rhs1=Dtrain*(Z+U)'; rhs=rho*temp_rhs1 + lambda_2*LLtWrefT;
                if rhoOld~=rho
                   A = rho*DDt + lambda_2*LLt;
                   [Q,R] = qr(A);
                end
                W = R\(Q'*rhs);
                
                W = W';
        
        end
        
        
        


         %% store values
                
        WD = W*Dtrain;
        WDval = W*Dval;
        [fcTrain, paraTrain, ~] = f.pLoss.getMisfit(WD, Ctrain);
        [fcVal, paraVal] = fVal.pLoss.getMisfit(WDval, Cval);
        
        accTrain = 100*(Ntrain-paraTrain(3))/Ntrain;
        accVal = 100*(Nval-paraVal(3))/Nval;
        
        
        
        %% keep weights containing highest accuracy and lowest misfit from validation set
        
        if fcVal<=lowestMisfit
        lowestMisfit = fcVal;
        wOptLoss = W;
        end
        if accVal>=highestAcc
        highestAcc = accVal;
        wOptAcc = W;
        end

        %% primal & dual residual
        
        
        

       
      
        
        his(iter,1) = iter; % current iter
        his(iter,2:3) = [fcTrain, fcVal]; % training and validation misfits
        his(iter,4:5) = [accTrain, accVal]; % training and validation accuracies
        
       
        his(iter,6) = currentRuntime;

        
        
        if currentRuntime>=stoppingCrit{2}
        his = his(1:iter,:);
        wFinal = W(:);
        break;
        end
        
        
        %% solve Z step (no validation)
        
        % create current Z regularizer
        
        
        rho_0=rho0/C;
        
        Zref = WD - U;
        pRegZ = tikhonovReg(opEye(size(Zref,1)*size(Zref,2)),rho_0, Zref(:));
        pLossZ = softmaxLossZ();
        
        fZ = classObjFctnZ(pLossZ,pRegZ,Ctrain);
        
        fZ.pLoss.addBias=0;
        
        optZ = newton();
        optZ.atol = atolZ;
        optZ.rtol = rtolZ;
        optZ.maxIter= maxIterZ;
        optZ.LS.maxIter=lsMaxIterZ;
        optZ.linSol.maxIter=linSolMaxIterZ;
        optZ.linSol.tol = linSolTolZ;
        optZ.out = outZ;
        
        Zold = Z;

      
        [Z, ~] = solve(optZ,fZ,Z(:));
        Z = reshape(Z, nc, Ntrain);
        
       
        
 
        
        
        
        %% update dual variable
        Uold=U;
        U = U + (Z - WD);
       
        
        iter = iter + 1;
        
end






tEndFunc=toc(tStarFunc)


end