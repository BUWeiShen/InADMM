classdef softmaxLossZ
    % classdef softmaxLoss
    %
    % object describing softmax loss function for Z inputs
    % (nClasses-by-nExamples)
    
    properties
       theta
       addBias
    end
   
    
    methods
        function this = softmaxLossZ(varargin)
            this.theta   = 1e-3;
            this.addBias = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        
        
        function [F,para,dF,d2F] = getMisfit(this,Z,C,varargin)
        % [F,para,dWF,d2WF,dYF,d2YF] = getMisfit(this,Z,C,varargin)
        %
        % Input:
        %  
        %   Z - 2D matrix, nClasses-by-nExamples
        %   C - 2D matrix, ground truth classes (nClasses-by-nExamples)
        %
        % Optional Input:
        %
        %   set via varargin
        %
        % Output:
        %
        %  F    - loss (average per example)
        %  para - vector of 3 values: unaveraged loss, nExamples, error
        %  dZF  - gradient of F wrt Z ((nClasses x nExamples)-by-1)
        %  d2ZF - Hessian of F wrt Z  (LinearOperator, (nClassesnExamples x nClassesxnExamples))
            
            doDF = (nargout>2); doD2F = (nargout>3);
            dF = []; d2F = []; 
            nex = size(C,2);
            
            Z = reshape(Z, [], nex);
            Z    = Z - max(Z,[],1);
            expZ    = exp(Z);
            
            Cp   = getLabels(this,expZ);
            err  = nnz(C-Cp)/2;
            F    = -sum(sum(C.*(Z))) + sum(log(sum(expZ,1)));
            para = [F,nex,err];
            F    = F/nex;

            
            if (doDF) && (nargout>=2)
               dF   = (1/nex)*vec(-C + expZ./sum(expZ,1)); 
            end
            
            if (doD2F) && (nargout>=3)
                matU  = @(U) reshape(U,size(expZ));
                
                d2Fmv = @(U) vec((expZ./sum(expZ,1)).*matU(U) - expZ.*sum(expZ.*matU(U),1)./sum(expZ,1).^2)/nex;
                
                szS = size(expZ);       
                d2F = LinearOperator(prod(szS),prod(szS),d2Fmv,d2Fmv);
            end
        end
        
        %%
        function [str,frmt] = hisNames(this)
            str  = {'F','accuracy'};
            frmt = {'%-12.2e','%-12.2f'};
        end
        function str = hisVals(this,para)
            str = [para(1)/para(2),(1-para(3)/para(2))*100];
        end       
        function [Cp,P] = getLabels(this,W,Y)
            S = W;
            nex = size(S,2);
            
            P      = S./sum(S,1);
            [~,jj] = max(P,[],1);
            Cp     = zeros(numel(P),1);
            ind    = sub2ind(size(P),jj(:),(1:nex)');
            Cp(ind)= 1;
            Cp     = reshape(Cp,size(P,1),[]);
        end
    end
    
end

