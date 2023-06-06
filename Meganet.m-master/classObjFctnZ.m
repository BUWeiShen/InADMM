classdef classObjFctnZ < objFctn
    % classdef classObjFctn < objFctn
    %
    % Objective function for classification,i.e., 
    %
    %   J(W) = loss(h(Z), C) + R(Z),
    %
    % where 
    % 
    %   Z    - weights of the classifier (having data taken into account)
    %   h    - hypothesis function
    %   Y    - features
    %   C    - class labels
    %   loss - loss function object
    %   R    - regularizer (object)
    
    properties
        pLoss
        pRegZ
        Y
        C
    end
    
    methods
        function this = classObjFctnZ(pLoss,pRegZ,C)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            
            this.pLoss  = pLoss;
            this.pRegZ  = pRegZ;
            this.C      = C;
            
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,Z,idx)
            if not(exist('idx','var')) || isempty(idx)
                C = this.C;
            else
                C = this.C(:,idx);
            end
            
            [Jc,hisLoss,dJ,H] = getMisfit(this.pLoss,Z,C);
            para = struct('F',Jc,'hisLoss',hisLoss);
            
            
            if not(isempty(this.pRegZ))
                [Rc,hisReg,dR,d2R] = regularizer(this.pRegZ,Z);
                para.hisReg = hisReg;
                para.Rc     = Rc;
                Jc = Jc + Rc; 
                dJ = vec(dJ)+ vec(dR);
                H = H + d2R;
                para.hisRZ = hisReg;
            end

            if nargout>4
                PC = opEye(numel(Z));
            end
        end
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            if not(isempty(this.pRegZ))
                [s,f] = hisNames(this.pRegZ);
                s{1} = [s{1} '(W)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = hisVals(this.pLoss,para.hisLoss);
            if not(isempty(this.pRegZ))
                his = [his, hisVals(this.pRegZ,para.hisRZ)];
            end
        end
        
        
        function str = objName(this)
            str = 'classObjFun';
        end

    end
end










