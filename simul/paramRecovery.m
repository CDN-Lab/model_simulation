%%%%%% paramRecovery.m %%%%%%

% This function uses the modelSimulations.m function, to generate n choice
% data sets, then fits both the Mazur hyperbolic discounting model (using
% the ITCMazur.m function) and the hyperbolic with nonlinear utility model
% (using the ITCnonlinear.m function). It builds a matrix with the
% recovered parameters for each model and its goodness of fit/model comp
% scores. 

% INPUT
% iterations    - a scalar that correspond to the n choice data sets to be
%                 simulated
% sourceModel   - the model used to generate data (this will be passed to
%                 modelSimulations.m), can be 'Mazur' or 'NL'
% OUTPUT
% recP          - a matrix (n,15) with all iterations of the simulation and
%                 the following information (order of columns): alpha;
%                 kappa; beta; NL fitted kappa; NL fitted
%                 beta; NL r2; NL AIC; NL BIC; NL LL; Maz fitted kappa; 
%                 Maz fitted beta; Maz r2; Maz AIC; Maz BIC; Maz LL 

function [recovParam] = paramRecovery(iterations,sourceModel)
recP = nan(iterations,27); 

    for n = 1:iterations
        [sData,alpha,kappa,beta] = modelSimulations(sourceModel);
        recP(n,1) = alpha; 
        recP(n,2) = kappa;
        recP(n,3) = beta;
        
        [info] = NLH(sData,alpha);
        
        recP(n,4) = info.b(2);
        recP(n,5) = info.b(1);
        recP(n,6) = info.r2;
        recP(n,7) = info.AIC;
        recP(n,8) = info.BIC;
        recP(n,9) = info.LL;
        
        [infoM] = LH(sData);
        
        recP(n,10) = infoM.b(2);
        recP(n,11) = infoM.b(1);
        recP(n,12) = infoM.r2;
        recP(n,13) = infoM.AIC;
        recP(n,14) = infoM.BIC;
        recP(n,15) = infoM.LL;
        
        [infoNE] = NLE(sData,alpha);
        
        recP(n,16) = infoNE.b(2);
        recP(n,17) = infoNE.b(1);
        recP(n,18) = infoNE.r2;
        recP(n,19) = infoNE.AIC;
        recP(n,20) = infoNE.BIC;
        recP(n,21) = infoNE.LL;
        
        [infoE] = LE(sData);
        
        recP(n,22) = infoE.b(2);
        recP(n,23) = infoE.b(1);
        recP(n,24) = infoE.r2;
        recP(n,25) = infoE.AIC;
        recP(n,26) = infoE.BIC;
        recP(n,27) = infoE.LL;
    end
  recovParam = array2table(recP,'VariableNames',{'alpha' 'kappa' 'beta'...
            'NLfitkappa' 'NLfitbeta' 'NL_r2' 'NL_AIC' 'NL_BIC' 'NL_LL'...
            'Mazfitkappa' 'Mazfitbeta' 'Maz_r2' 'Maz_AIC' 'Maz_BIC' 'Maz_LL'...
            'NLEfitkappa' 'NLEfitbeta' 'NLE_r2' 'NLE_AIC' 'NLE_BIC' 'NLE_LL'...
            'LEfitkappa' 'LEfitbeta' 'LE_r2' 'LE_AIC' 'LE_BIC' 'LE_LL'});  
end

