

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
maxIter = 10000;
DDt = Dtrain*Dtrain';
LLt = L*L';

Wref    = zeros(nc,nf);
LLtWref = LLt*Wref';
A = rho0*DDt + lambda_2*LLt; %fixed rho


pRegW   = tikhonovReg(Lout,lambda_2);
pLoss   = softmaxLossZ();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fVal   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=0; fVal.pLoss.addBias=0;

%% InADMM parameters
normD=normest(Dtrain,0.0001);
inexact_mu=sqrt(2*rho0)*normD;
inexact_tau=1/sqrt(lambda_2*min(eig(L'*L)));
inexact_sigma=0.99*1/(1+inexact_mu*inexact_tau);

inMu=inexact_mu;
inTau=1/sqrt(lambda_2*min(eig(L'*L)));
proposed_sigma=0.99*1/(1+inMu*inTau);

%% initial admm values

w0      = zeros(nc,nf);
z0       = w0*Dtrain; 
u0   = zeros(nc,Ntrain);


 





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
% model parameters
param.C               = C;
param.lambda_2        = lambda_2;
param.rho0            = rho0;
param.addBias         = addBias;
param.Wref            = Wref;
% initail points
param.W               = w0;
param.U               = u0;
param.Z               = z0;
% dataset
param.Dtrain          = Dtrain;
param.Dval            = Dval;
param.Ctrain          = Ctrain;
param.Cval            = Cval;
% matrix and class obj functions
param.L               = L;
param.f               = f; % class obj func with training data
param.fVal            = fVal; % class obj func with val data
param.A               = A;
param.LLtWref         = LLtWref;
% InADMM parameters
param.inMu            = inMu;
param.sigma_another   = inexact_sigma;
param.sigma           = proposed_sigma;
% Stopping parameters
param.maxIter         = maxIter;
param.stoppingCrit    = stoppingCrit;
% z-step (i.e. alpha-update) parameters
param.atolZ           = atolZ;
param.rtolZ           = rtolZ; 
param.maxIterZ        = maxIterZ;
param.linSolMaxIterZ  = linSolMaxIterZ;
param.linSolTolZ      = linSolTolZ;
param.lsMaxIterZ      = lsMaxIterZ;
param.outZ            = outZ;

%% misfit & accuracy of initial points
WD = w0*Dtrain;
WDval = w0*Dval;
[fcTrain0, paraTrain0, ~] = f.pLoss.getMisfit(WD, Ctrain);
[fcVal0, paraVal0] = fVal.pLoss.getMisfit(WDval, Cval);
accTrain0 = 100*(Ntrain-paraTrain0(3))/Ntrain;
accVal0 = 100*(Nval-paraVal0(3))/Nval;

fprintf('\niter\tfTrain\t fVal\t trainAcc valAcc \n')
fprintf('%d\t%1.2e %1.2e %1.2e\t%1.2e\n',...
        0, fcTrain0,fcVal0, accTrain0, accVal0);

%% Train with specific LeastSquares solver
lsSolver = 'fix_tol_pcg'; % 'cholesky', or 'qr','fix_tol_pcg', 'proposed_in_pcg','another_in_pcg'
fix_pcg_tol = 1e-2;

param.lsSolver        = lsSolver;
param.fix_pcg_tol=fix_pcg_tol;

[wFinal, wOptLoss, wOptAcc, hisOpt] = InADMM(param);
hisFix2Pcg=hisOpt;
%% check validation accuracy
WDLoss = wOptLoss*Dval;

[FcValLoss, paraValLoss] = fVal.pLoss.getMisfit(WDLoss, Cval);
errValLoss = paraValLoss(3);
accValLoss = 100*(Nval-errValLoss)/Nval;

% save best validation accuracy values
% WDAcc = reshape(wOptAcc, nc, [])*Dval;
WDAcc = wOptAcc*Dval;

[FcValAcc, paraValAcc] = fVal.pLoss.getMisfit(WDAcc, Cval);
errValAcc = paraValAcc(3);
accValAcc = 100*(Nval-errValAcc)/Nval;

fprintf('\n ------ VALIDATION RESULTS ------ \n')
fprintf('\n wOptLoss: fVal = %1.2e, accVal = %1.2f\n', FcValLoss, accValLoss);
fprintf('\n wOptAcc: fVal = %1.2e, accVal = %1.2f\n', FcValAcc, accValAcc);


%% check testing accuracy


pLossTest = softmaxLoss();
%fTest     = classObjFctn(pLossTest,pRegW,Dtest(1:end-1,:),Ctest);
fTest     = classObjFctn(pLossTest,pRegW,Dtest,Ctest); fTest.pLoss.addBias=0;
% weights that minimize validation misfit
[~, paraTestLoss] = fTest.eval(wOptLoss(:));


FcTestLoss=paraTestLoss.F;
errTestLoss = paraTestLoss.hisLoss(3);
accTestLoss = 100*(Ntest-errTestLoss)/Ntest;

% weights that maximize validation accuracy
[~, paraTestAcc] = fTest.eval(wOptAcc(:));
FcTestAcc=paraTestAcc.F;
errTestAcc = paraTestAcc.hisLoss(3);
accTestAcc = 100*(Ntest-errTestAcc)/Ntest;

fprintf('\n ------ TESTING RESULTS ------ \n')
fprintf('\n wOptLoss: fTest = %1.2e, accTest = %1.2f\n', FcTestLoss, accTestLoss);
fprintf('\n wOptAcc: fTest = %1.2e, accTest = %1.2f\n', FcTestAcc, accTestAcc);

save('fix_tol_1e_2-400Results.mat', 'hisOpt', 'wOptLoss', 'wOptAcc', 'wFinal', 'lambda_2', 'atol', 'rtol', 'atolZ', 'rtolZ', 'linSolMaxIterZ', 'lsMaxIterZ', 'maxIterZ', 'rho0', 'FcTestLoss', 'accTestLoss', 'FcTestAcc', 'accTestAcc');
    


%% Train with specific LeastSquares solver
lsSolver = 'fix_tol_pcg'; % 'cholesky', or 'qr','fix_tol_pcg', 'proposed_in_pcg','another_in_pcg'
fix_pcg_tol = 1e-4;

param.lsSolver        = lsSolver;
param.fix_pcg_tol=fix_pcg_tol;

[wFinal, wOptLoss, wOptAcc, hisOpt] = InADMM(param);
hisFix4Pcg=hisOpt;
%% check validation accuracy
WDLoss = wOptLoss*Dval;

[FcValLoss, paraValLoss] = fVal.pLoss.getMisfit(WDLoss, Cval);
errValLoss = paraValLoss(3);
accValLoss = 100*(Nval-errValLoss)/Nval;

% save best validation accuracy values
% WDAcc = reshape(wOptAcc, nc, [])*Dval;
WDAcc = wOptAcc*Dval;

[FcValAcc, paraValAcc] = fVal.pLoss.getMisfit(WDAcc, Cval);
errValAcc = paraValAcc(3);
accValAcc = 100*(Nval-errValAcc)/Nval;

fprintf('\n ------ VALIDATION RESULTS ------ \n')
fprintf('\n wOptLoss: fVal = %1.2e, accVal = %1.2f\n', FcValLoss, accValLoss);
fprintf('\n wOptAcc: fVal = %1.2e, accVal = %1.2f\n', FcValAcc, accValAcc);


%% check testing accuracy


pLossTest = softmaxLoss();
%fTest     = classObjFctn(pLossTest,pRegW,Dtest(1:end-1,:),Ctest);
fTest     = classObjFctn(pLossTest,pRegW,Dtest,Ctest); fTest.pLoss.addBias=0;
% weights that minimize validation misfit
[~, paraTestLoss] = fTest.eval(wOptLoss(:));


FcTestLoss=paraTestLoss.F;
errTestLoss = paraTestLoss.hisLoss(3);
accTestLoss = 100*(Ntest-errTestLoss)/Ntest;

% weights that maximize validation accuracy
[~, paraTestAcc] = fTest.eval(wOptAcc(:));
FcTestAcc=paraTestAcc.F;
errTestAcc = paraTestAcc.hisLoss(3);
accTestAcc = 100*(Ntest-errTestAcc)/Ntest;

fprintf('\n ------ TESTING RESULTS ------ \n')
fprintf('\n wOptLoss: fTest = %1.2e, accTest = %1.2f\n', FcTestLoss, accTestLoss);
fprintf('\n wOptAcc: fTest = %1.2e, accTest = %1.2f\n', FcTestAcc, accTestAcc);

save('fix_tol_1e_4-400Results.mat', 'hisOpt', 'wOptLoss', 'wOptAcc', 'wFinal', 'lambda_2', 'atol', 'rtol', 'atolZ', 'rtolZ', 'linSolMaxIterZ', 'lsMaxIterZ', 'maxIterZ', 'rho0', 'FcTestLoss', 'accTestLoss', 'FcTestAcc', 'accTestAcc');
  

%% Train with specific LeastSquares solver
lsSolver = 'fix_tol_pcg'; % 'cholesky', or 'qr','fix_tol_pcg', 'proposed_in_pcg','another_in_pcg'
fix_pcg_tol = 1e-6;

param.lsSolver        = lsSolver;
param.fix_pcg_tol=fix_pcg_tol;

[wFinal, wOptLoss, wOptAcc, hisOpt] = InADMM(param);
hisFix6Pcg=hisOpt;
%% check validation accuracy
WDLoss = wOptLoss*Dval;

[FcValLoss, paraValLoss] = fVal.pLoss.getMisfit(WDLoss, Cval);
errValLoss = paraValLoss(3);
accValLoss = 100*(Nval-errValLoss)/Nval;

% save best validation accuracy values
% WDAcc = reshape(wOptAcc, nc, [])*Dval;
WDAcc = wOptAcc*Dval;

[FcValAcc, paraValAcc] = fVal.pLoss.getMisfit(WDAcc, Cval);
errValAcc = paraValAcc(3);
accValAcc = 100*(Nval-errValAcc)/Nval;

fprintf('\n ------ VALIDATION RESULTS ------ \n')
fprintf('\n wOptLoss: fVal = %1.2e, accVal = %1.2f\n', FcValLoss, accValLoss);
fprintf('\n wOptAcc: fVal = %1.2e, accVal = %1.2f\n', FcValAcc, accValAcc);


%% check testing accuracy


pLossTest = softmaxLoss();
%fTest     = classObjFctn(pLossTest,pRegW,Dtest(1:end-1,:),Ctest);
fTest     = classObjFctn(pLossTest,pRegW,Dtest,Ctest); fTest.pLoss.addBias=0;
% weights that minimize validation misfit
[~, paraTestLoss] = fTest.eval(wOptLoss(:));


FcTestLoss=paraTestLoss.F;
errTestLoss = paraTestLoss.hisLoss(3);
accTestLoss = 100*(Ntest-errTestLoss)/Ntest;

% weights that maximize validation accuracy
[~, paraTestAcc] = fTest.eval(wOptAcc(:));
FcTestAcc=paraTestAcc.F;
errTestAcc = paraTestAcc.hisLoss(3);
accTestAcc = 100*(Ntest-errTestAcc)/Ntest;

fprintf('\n ------ TESTING RESULTS ------ \n')
fprintf('\n wOptLoss: fTest = %1.2e, accTest = %1.2f\n', FcTestLoss, accTestLoss);
fprintf('\n wOptAcc: fTest = %1.2e, accTest = %1.2f\n', FcTestAcc, accTestAcc);

save('fix_tol_1e_6-400Results.mat', 'hisOpt', 'wOptLoss', 'wOptAcc', 'wFinal', 'lambda_2', 'atol', 'rtol', 'atolZ', 'rtolZ', 'linSolMaxIterZ', 'lsMaxIterZ', 'maxIterZ', 'rho0', 'FcTestLoss', 'accTestLoss', 'FcTestAcc', 'accTestAcc');


%% Train with specific LeastSquares solver
lsSolver = 'proposed_in_pcg'; % 'cholesky', or 'qr','fix_tol_pcg', 'proposed_in_pcg','another_in_pcg'
param.lsSolver        = lsSolver;
stoppingCrit{1} = 'runtime'; stoppingCrit{2} = 300;
param.stoppingCrit    = stoppingCrit;
[wFinal, wOptLoss, wOptAcc, hisOpt] = InADMM(param);
hisPropInPCG=hisOpt;

%% check validation accuracy
WDLoss = wOptLoss*Dval;

[FcValLoss, paraValLoss] = fVal.pLoss.getMisfit(WDLoss, Cval);
errValLoss = paraValLoss(3);
accValLoss = 100*(Nval-errValLoss)/Nval;

% save best validation accuracy values
% WDAcc = reshape(wOptAcc, nc, [])*Dval;
WDAcc = wOptAcc*Dval;

[FcValAcc, paraValAcc] = fVal.pLoss.getMisfit(WDAcc, Cval);
errValAcc = paraValAcc(3);
accValAcc = 100*(Nval-errValAcc)/Nval;

fprintf('\n ------ VALIDATION RESULTS ------ \n')
fprintf('\n wOptLoss: fVal = %1.2e, accVal = %1.2f\n', FcValLoss, accValLoss);
fprintf('\n wOptAcc: fVal = %1.2e, accVal = %1.2f\n', FcValAcc, accValAcc);


%% check testing accuracy


pLossTest = softmaxLoss();
%fTest     = classObjFctn(pLossTest,pRegW,Dtest(1:end-1,:),Ctest);
fTest     = classObjFctn(pLossTest,pRegW,Dtest,Ctest); fTest.pLoss.addBias=0;
% weights that minimize validation misfit
[~, paraTestLoss] = fTest.eval(wOptLoss(:));


FcTestLoss=paraTestLoss.F;
errTestLoss = paraTestLoss.hisLoss(3);
accTestLoss = 100*(Ntest-errTestLoss)/Ntest;

% weights that maximize validation accuracy
[~, paraTestAcc] = fTest.eval(wOptAcc(:));
FcTestAcc=paraTestAcc.F;
errTestAcc = paraTestAcc.hisLoss(3);
accTestAcc = 100*(Ntest-errTestAcc)/Ntest;

fprintf('\n ------ TESTING RESULTS ------ \n')
fprintf('\n wOptLoss: fTest = %1.2e, accTest = %1.2f\n', FcTestLoss, accTestLoss);
fprintf('\n wOptAcc: fTest = %1.2e, accTest = %1.2f\n', FcTestAcc, accTestAcc);

save('prop_inADMM-400Results.mat', 'hisOpt', 'wOptLoss', 'wOptAcc', 'wFinal', 'lambda_2', 'atol', 'rtol', 'atolZ', 'rtolZ', 'linSolMaxIterZ', 'lsMaxIterZ', 'maxIterZ', 'rho0', 'FcTestLoss', 'accTestLoss', 'FcTestAcc', 'accTestAcc');
  

%% Train with specific LeastSquares solver
lsSolver = 'another_in_pcg'; % 'cholesky', or 'qr','fix_tol_pcg', 'proposed_in_pcg','another_in_pcg'
param.lsSolver        = lsSolver;
stoppingCrit{1} = 'runtime'; stoppingCrit{2} = 400;
param.stoppingCrit    = stoppingCrit;
[wFinal, wOptLoss, wOptAcc, hisOpt] = InADMM(param);
hisPropInPCG2=hisOpt;

%% check validation accuracy
WDLoss = wOptLoss*Dval;

[FcValLoss, paraValLoss] = fVal.pLoss.getMisfit(WDLoss, Cval);
errValLoss = paraValLoss(3);
accValLoss = 100*(Nval-errValLoss)/Nval;

% save best validation accuracy values
% WDAcc = reshape(wOptAcc, nc, [])*Dval;
WDAcc = wOptAcc*Dval;

[FcValAcc, paraValAcc] = fVal.pLoss.getMisfit(WDAcc, Cval);
errValAcc = paraValAcc(3);
accValAcc = 100*(Nval-errValAcc)/Nval;

fprintf('\n ------ VALIDATION RESULTS ------ \n')
fprintf('\n wOptLoss: fVal = %1.2e, accVal = %1.2f\n', FcValLoss, accValLoss);
fprintf('\n wOptAcc: fVal = %1.2e, accVal = %1.2f\n', FcValAcc, accValAcc);


%% check testing accuracy


pLossTest = softmaxLoss();
%fTest     = classObjFctn(pLossTest,pRegW,Dtest(1:end-1,:),Ctest);
fTest     = classObjFctn(pLossTest,pRegW,Dtest,Ctest); fTest.pLoss.addBias=0;
% weights that minimize validation misfit
[~, paraTestLoss] = fTest.eval(wOptLoss(:));


FcTestLoss=paraTestLoss.F;
errTestLoss = paraTestLoss.hisLoss(3);
accTestLoss = 100*(Ntest-errTestLoss)/Ntest;

% weights that maximize validation accuracy
[~, paraTestAcc] = fTest.eval(wOptAcc(:));
FcTestAcc=paraTestAcc.F;
errTestAcc = paraTestAcc.hisLoss(3);
accTestAcc = 100*(Ntest-errTestAcc)/Ntest;

fprintf('\n ------ TESTING RESULTS ------ \n')
fprintf('\n wOptLoss: fTest = %1.2e, accTest = %1.2f\n', FcTestLoss, accTestLoss);
fprintf('\n wOptAcc: fTest = %1.2e, accTest = %1.2f\n', FcTestAcc, accTestAcc);

save('another_inADMM-400Results.mat', 'hisOpt', 'wOptLoss', 'wOptAcc', 'wFinal', 'lambda_2', 'atol', 'rtol', 'atolZ', 'rtolZ', 'linSolMaxIterZ', 'lsMaxIterZ', 'maxIterZ', 'rho0', 'FcTestLoss', 'accTestLoss', 'FcTestAcc', 'accTestAcc');
  
%% Graph
g_compareCIFAR=figure;
subplot(2,2,1)
t=plot([0;hisPropInPCG(:,6)],[fcTrain0;hisPropInPCG(:,2)], 'b-',[0;hisPropInPCG2(:,6)],[fcTrain0;hisPropInPCG2(:,2)], 'b-.',...
    [0;hisFix2Pcg(:,6)],[fcTrain0;hisFix2Pcg(:,2)], 'r:*',[0;hisFix4Pcg(:,6)], [fcTrain0;hisFix4Pcg(:,2)], 'c--',...
    [0;hisFix6Pcg(:,6)],[fcVal0;hisFix6Pcg(:,2)], 'k:', 'MarkerSize', 8,'LineWidth', 1);
axis([0 400 0 2.5])
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('Proposed I','Proposed II','1e-2','1e-4','1e-6');
xlabel('Time')
title('Train Misfit')
subplot(2,2,2)
t=plot([0;hisPropInPCG(:,6)],[fcVal0;hisPropInPCG(:,3)], 'b-',[0;hisPropInPCG2(:,6)],[fcVal0;hisPropInPCG2(:,3)], 'b-.',...
    [0;hisFix2Pcg(:,6)],[fcVal0;hisFix2Pcg(:,3)], 'r:*',[0;hisFix4Pcg(:,6)], [fcVal0;hisFix4Pcg(:,3)], 'c--',...
    [0;hisFix6Pcg(:,6)],[fcVal0;hisFix6Pcg(:,3)], 'k:','MarkerSize', 8,'LineWidth', 1);
axis([0 400 0 2.5])
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('Proposed I','Proposed II','1e-2','1e-4','1e-6');
xlabel('Time')
title('Validation Misfit')
subplot(2,2,3)
t=plot([0;hisPropInPCG(:,6)],[accTrain0/100;hisPropInPCG(:,4)/100], 'b-',[0;hisPropInPCG2(:,6)],[accTrain0/100;hisPropInPCG2(:,4)/100], 'b-.',...
    [0;hisFix2Pcg(:,6)],[accTrain0/100;hisFix2Pcg(:,4)/100], 'r:*',[0;hisFix4Pcg(:,6)], [accTrain0/100;hisFix4Pcg(:,4)/100], 'c--',...
    [0;hisFix6Pcg(:,6)],[accTrain0/100;hisFix6Pcg(:,4)/100], 'k:','MarkerSize', 8,'LineWidth', 1);
axis([0 400 0 1])
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('Proposed I','Proposed II','1e-2','1e-4','1e-6','Location','southeast');
xlabel('Time')
title('Train Accuracy')
subplot(2,2,4)
t=plot([0;hisPropInPCG(:,6)],[accVal0/100;hisPropInPCG(:,5)/100], 'b-',[0;hisPropInPCG2(:,6)],[accVal0/100;hisPropInPCG2(:,5)/100], 'b-.',...
    [0;hisFix2Pcg(:,6)],[accVal0/100;hisFix2Pcg(:,5)/100], 'r:*',[0;hisFix4Pcg(:,6)], [accVal0/100;hisFix4Pcg(:,5)/100], 'c--',...
    [0;hisFix6Pcg(:,6)],[accVal0/100;hisFix6Pcg(:,5)/100], 'k:','MarkerSize', 8,'LineWidth', 1);
axis([0 400 0 1])
xlabel('Time')
set(t(2), 'LineWidth', 1.5) 
legend('Proposed I','Proposed II','1e-2','1e-4','1e-6','Location','southeast');
xlabel('Time')
title('Validation Accuracy')