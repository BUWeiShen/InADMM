classdef lbfgsSoftmax < optimizer
    % classdef lbfgs < optimizer
    %
    % lBFGS scheme for minimizing non-linear objective
    
    properties
        maxIter
        atol
        rtol
        maxStep
        out
        LS
        linSol
        maxlBFGS
        stoppingTime % time-based stopping criteria (seconds)
    end
    
    methods
        
        function this = lbfgsSoftmax(varargin)
            this.maxIter = 10;
            this.atol    = 1e-3;
            this.rtol    = 1e-3;
            this.maxStep = 1.0;
            this.out     = 0;
            this.linSol  = steihaugPCG('tol',1e-1,'maxIter',10);
            this.LS      = Armijo();
            this.maxlBFGS   = 10;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end;
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|','iterCG','relresCG','mu','LS'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e','%-12d','%-12.2e','%-12.2e','%-12d'};
        end
        
%         function [xc,His] = solve(this,fctn,xc,fval)
        function [xc, xOptLoss, xOptAcc,His] = solve(this,fctn,xc,fval)
            if not(exist('fval','var')); fval = []; end;
            
            [str,frmt] = hisNames(this);
            
            % parse objective functions
            [fctn,objFctn,objNames,objFrmt,objHis]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt,obj2His] = parseObjFctn(this,fval);
            str = [str,obj2Names{:}]; frmt = [frmt,obj2Frmt{:}];
            doVal     = not(isempty(obj2Fctn));
            
            % evaluate training and validation
            [Jc,para,dJ,H0,PC] = fctn(xc); pVal = [];
            if doVal
                if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                    [Fval,pVal] = fval([xc; para.W(:)]);
                else
                    [Fval,pVal] = fval(xc);
                end
            end
            
            if this.out>0
                fprintf('== lbfgs (n=%d,maxIter=%d,maxStep=%1.1e) ===\n',...
                    numel(xc), this.maxIter, this.maxStep);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            
%             his = zeros(1,numel(str));
            his = zeros(1,numel(str)+1); % add one column for timing
            nrm0 = norm(dJ(:));
            zBFGS = [];     % memory for BFGS gradient directions
            sBFGS = [];     % memory for BFGS directions
            nBFGS  = 0;     % counter for number of limited BFGS directions
            
            iter = 1;
            xOld = xc;
            
            currentRuntime = 0; 
            lowestMisfit = Inf;
            lowestClassError   = Inf; 
            
            while iter <= this.maxIter
                
%                 if this.stoppingTime < currentRuntime, break; end
                
                tCurrentIter = tic;
                if iter>1
                    zz = dJ - dJold;
                    ss =  xc - xOld;
                    if zz'*ss > 0
                        start = 2-(nBFGS<this.maxlBFGS);
                        zBFGS = [zBFGS(:,start:end),zz];
                        sBFGS = [sBFGS(:,start:end),ss];
                        nBFGS = min(this.maxlBFGS,nBFGS+1);
                    else
                        warning('   y''*s < 0, skip update');
                    end
                end
                
                his(iter,1:4)  = [iter,gather(Jc),gather(norm(xOld(:)-xc(:))),gather(norm(dJ(:))/nrm0)];
                if this.out>0
                    fprintf([frmt{1:4}], his(iter,1:4));
                end
                if (norm(dJ(:))/nrm0 < this.rtol) || (norm(dJ(:))< this.atol), break; end
                
                % solve the linear system
                [s,flag,relresCG,iterCG,resvec] = bfgsrec(this,nBFGS,sBFGS,zBFGS,H0,PC,-dJ(:));
                if norm(s) == 0, s = dJ(:)/norm(dJ(:)); end
                clear H0
                
                his(iter,5:6) = [iterCG, relresCG];
                if this.out>0
                    fprintf([frmt{5:6}], his(iter,5:6));
                end
                
                % s = d2F\(-dF(:));
                if max(abs(s(:))) > this.maxStep
                    s = s/max(abs(s(:))) * this.maxStep;
                end
                % line search
                if iter == 1; mu = 1.0; end
                [xt,mu,lsIter] = lineSearch(this.LS,fctn,xc,mu,s,Jc,dJ);
                if (lsIter > this.LS.maxIter)
                    disp('LSB in lbfgs'); %keyboard
                    his = his(1:iter,:);
                    break;
                end
                his(iter,7:8) = [mu lsIter];
                if this.out>0
                    fprintf([frmt{7:8}], his(iter,7:8));
                end
                if lsIter == 1
                    mu = min(mu*1.5,1);
                end
                xOld       = xc;
                xc         = xt;
                if doVal
                    if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                        [Fval,pVal] = fval([xc; para.W(:)]);
                    else
                        [Fval,pVal] = fval(xc);
                    end
                    
                    % keep weights containing highest accuracy and lowest misfit from validation set
                    if pVal.F<lowestMisfit
                        lowestMisfit = pVal.F;
                        xOptLoss = xc;
                    end
                    if pVal.hisLoss(3)<lowestClassError
                        lowestClassError = pVal.hisLoss(3);
                        xOptAcc = xc;
                    end
                    
                end
                
                dJold = dJ;
                [Jc,para,dJ,H0] = fctn(xc);
                
                tCurrentIter = toc(tCurrentIter);
                currentRuntime = currentRuntime + tCurrentIter;
                
                if size(his,2)>=9
%                     his(iter,9:end) = [gather(objHis(para)), gather(obj2His(pVal))];
                    his(iter,9:end) = [gather(objHis(para)), gather(obj2His(pVal)), currentRuntime];
                    if this.out>0
                        fprintf([frmt{9:end}],his(iter,9:end));
                    end
                end
                if this.out>0
                    fprintf('\n');
                end
                
                if this.stoppingTime < currentRuntime
                    fprintf('\nstop reason: time\n')
                    his = his(1:iter,:);
                    break; 
                end
                
                iter = iter + 1;
            end
            His = struct('str',{str},'frmt',{frmt},'his',his(1:min(iter,this.maxIter),:));
        end
        function[d,flag,relresCG,iterCG,resvec] = bfgsrec(this,n,S,Z,H,PC,d)
            if n == 0
                [d,flag,relresCG,iterCG,resvec] = solve(this.linSol,H,d(:),[],PC);
            else
                alpha = (S(:,n)'*d)/(Z(:,n)'*S(:,n));
                d     = d - alpha*Z(:,n);
                [d,flag,relresCG,iterCG,resvec]     = bfgsrec(this,n-1,S,Z,H,PC,d);
                d     = d + (alpha - (Z(:,n)'*d)/(Z(:,n)'*S(:,n)))*S(:,n);
            end
        end
    end
end