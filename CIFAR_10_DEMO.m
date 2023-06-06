

%load('CIFARdata_normalized.mat')
addpath(genpath('~/ADMMSoftmaxCode/'));

N = 50000; Ntrain = 0.8*N; Nval = 0.2*N; Ntest = 0.2*N;

layer = 'pool5';
[Dtrain,Ctrain,Dval,Cval,Dtest,Ctest] = setupCIFAR10Vgg16(N, Ntest, layer);
Dtrain = double(Dtrain); Dtest = double(Dtest); Dval = double(Dval);
nf = size(Dtrain,1); 
nc = size(Ctrain,1);

fprintf('maxDtrain = %1.2e, minDtrain = %1.2e\n', max(Dtrain(:)), min(Dtrain(:)));
fprintf('maxDval = %1.2e, minDval = %1.2e\n', max(Dval(:)), min(Dval(:)));

Dtrain    = normalizeData(Dtrain, size(Dtrain,1));
Dval      = normalizeData(Dval, size(Dval,1));
Dtest     = normalizeData(Dtest, size(Dtest,1));
fprintf('maxDtrain = %1.2e, minDtrain = %1.2e\n', max(Dtrain(:)), min(Dtrain(:)));
fprintf('maxDval = %1.2e, minDval = %1.2e\n', max(Dval(:)), min(Dval(:)));


%% regularization
C=nf*nf*10;
rho0=0.05;
lambda_2 = 1;
addBias=true;
nImg = [7 7]; channelsOut = 512; % vgg16  pool 5
% smoothness reg. operator
fprintf('using smoothness! reg. operator...\n')
Ltemp = getLaplacian(nImg, 1./nImg);

L = genBlkDiag(Ltemp,channelsOut-1);

%  add bias to laplace operator 
if addBias==true
    L = sparse([L zeros(size(L,1),1); zeros(1,size(L,2)) 1]);
end

Lout = sparse(genBlkDiag(L, nc-1));

fprintf('size of Lout = %d = %1.2e...\n', size(Lout,1))
fprintf(' max of Lout... ');
max(Lout(:))

size(Dtrain)
%% start optimization
DDt = Dtrain*Dtrain';
LLt = L*L';

Wref    = zeros(nc,nf);
LLtWref = LLt*Wref';


pRegW   = tikhonovReg(Lout,lambda_2);
pLoss   = softmaxLossZ();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fVal   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=0; fVal.pLoss.addBias=0;

%% initial admm values

w0      = zeros(nc,nf);
Wref    = zeros(nc,nf);
z0       = w0*Dtrain; 
u0   = zeros(nc,Ntrain);

A = rho0*DDt + lambda_2*LLt; %fixed rho

maxIter = 10000; atol = 1e-12; rtol = 1e-12;
out=1; varRho=0; scaleRho = 2; mu = 10;
rhoLowerBound = 1e-16;
rhoUpperBound = 1e3;

normD=normest(Dtrain,0.0001);
inexact_mu=sqrt(2*rho0)*normD;
inexact_tau=1/sqrt(lambda_2*min(eig(L'*L)));
inexact_sigma=0.99*1/(1+inexact_mu*inexact_tau);

inMu=inexact_mu;
inTau=1/sqrt(lambda_2*min(eig(L'*L)));
proposed_sigma=0.99*1/(1+inMu*inTau);


%% LeastSquares solver
lsSolver = 'proposed_in_pcg'; % 'cholesky', or 'qr','fix_tol_pcg'
fix_pcg_tol = 1e-2;


%% Z-step parameters
maxIterZ = 100; % max number of Z newton iters
linSolMaxIterZ = 50; % max number of CG iters per newton step in Z step
lsMaxIterZ= 20; % max number of linesearch armijo iters per lin sol in Z step
atolZ = 1e-16; rtolZ=1e-16; % abs and rel tolerance for z solve
outZ = 0; % output for Z solve
linSolTolZ = 1e-16; % tolerance of linear solver (steihaug CG) for Z newton step
%% stopping criteria

stoppingCrit{1} = 'runtime'; stoppingCrit{2} = 400; % stop after 400 seconds


%% setup param structure
param.C=C;
param.fix_pcg_tol=fix_pcg_tol;
param.W=w0;
param.U=u0;
param.Z=z0;
param.inMu=inMu;
param.A= A;
param.LLtWref = LLtWref;
param.sigma_another=inexact_sigma;
param.sigma=proposed_sigma;
param.maxIter         = maxIter;
param.stoppingCrit    = stoppingCrit;
param.varRho          = varRho;
param.rhoLowerBound   = rhoLowerBound;
param.rhoUpperBound   = rhoUpperBound;
param.mu              = mu;
param.atol            = atol;
param.rtol            = rtol;

param.lambda_2        = lambda_2;
param.lsSolver        = lsSolver;
param.addBias         = addBias;
param.rho0            = rho0;
param.scaleRho        = scaleRho;
param.out             = out;


param.Wref            = Wref;
param.Dtrain          = Dtrain;
param.Dval            = Dval;
param.Ctrain          = Ctrain;
param.Cval            = Cval;
param.L               = L;

param.f               = f; % class obj func with training data
param.fVal            = fVal; % class obj func with val data

% z parameters
param.atolZ           = atolZ;
param.rtolZ           = rtolZ; 
param.maxIterZ        = maxIterZ;
param.linSolMaxIterZ  = linSolMaxIterZ;
param.linSolTolZ      = linSolTolZ;
param.lsMaxIterZ      = lsMaxIterZ;
param.outZ            = outZ;

%% train


WD = w0*Dtrain;
WDval = w0*Dval;
[fcTrain0, paraTrain0, ~] = f.pLoss.getMisfit(WD, Ctrain);
[fcVal0, paraVal0] = fVal.pLoss.getMisfit(WDval, Cval);
accTrain0 = 100*(Ntrain-paraTrain0(3))/Ntrain;
accVal0 = 100*(Nval-paraVal0(3))/Nval;

fprintf('\niter\tfTrain\t fVal\t trainAcc valAcc \n')
fprintf('%d\t%1.2e %1.2e %1.2e\t%1.2e\n',...
        0, fcTrain0,fcVal0, accTrain0, accVal0);
        


[wFinal, wOptLoss, wOptAcc, hisOpt] = InADMM(param);
