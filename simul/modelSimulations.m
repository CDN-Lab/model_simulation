%%%% modelSimulations.m %%%%%%%
% This function simulates choice data for a given model (Mazur or
% Hyperbolic with nonlinear utility
% Silvia 02.25.16

% INPUT   
% model   -'LH'     for Linear utility Hyperbolic Discounting Model (Mazur)
%         -'NLH'    for Nonlinear Utility Hyperbolic Discounting Model
%         -'LE'     for Linear utility Exponential Discounting Model
%         -'NLE'    for Nonlinear Utility Exponential Discounting Model
% OUTPUT  
% sData   - sData, a vector of 1s and 0s, 1 for later chosen, 0 for now
%           chosen

function [sData,alpha,kappa,beta] = modelSimulations(model)

alpha = (2-0.112)*rand + 0.112;
kappa = (6.4-0.001)*rand + 0.001;
beta = (3-0)*rand + (0);
% alpha = (10-0.0001)*rand + 0.112;
% kappa = (0.7-0.0001)*rand + 0.0001;
% beta = (10-(0.1))*rand + (0.001);
addpath('/Users/silvia/Dropbox (Personal)/MATLAB/ITC/Analysis/group/simulations');
load 'choiceSet.mat'
vNow = choiceSet(:,1);
vLater = choiceSet(:,2);
dNow = 0;
dLater = choiceSet(:,3);

if strcmp(model,'LH');
    uNow = (vNow)./(1+kappa*dNow);
    uLater = (vLater)./(1+kappa*dLater);
    pLater = 1 ./ (1 + exp(-beta*(uNow-uLater)));
elseif strcmp(model,'NLH')
    uNow = (vNow.^alpha)./(1+kappa*dNow);
    uLater = (vLater.^alpha)./(1+kappa*dLater);
    pLater = 1 ./ (1 + exp(-beta*(uNow-uLater)));
elseif strcmp(model,'LE')
    uNow = (vNow).*exp(-kappa*dNow);
    uLater = (vLater).*exp(-kappa*dLater);
    pLater = 1 ./ (1 + exp(-beta*(uNow-uLater)));
elseif strcmp(model,'NLE')
    uNow = (vNow.^alpha).*exp(-kappa*dNow);
    uLater = (vLater.^alpha).*exp(-kappa*dLater);
    pLater = 1 ./ (1 + exp(-beta*(uNow-uLater)));
end

draw = rand(102,1);
sData = draw > pLater;

end

