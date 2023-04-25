%% recoveryAnalysis

% Script to generate and fit data with CASANDRE process model. In the paper
% associated with CASADRE we explored the recoverability of model parameters
% with changing the unique number of stimulus values (parameter: stimValue)
% and repetitions per stimulus value (parameter: stimReps).

% Parameters used to generate figure 4A & C: 
% guessRate   = 0;
% stimSens    = 1;
% stimCrit    = 0
% uncMeta     = [0.2 04 0.8 1.6 3.2];
% confCrit    = 0.75;
% asymFlag    = 0;

close all;
clearvars;
clc;

% Set experiment parameters
stimValue = linspace(-3, 3, 11);   % The different stimulus conditions in units of stimulus magnitude (e.g., orientation in degrees)
stimReps  = 200;                   % The number of repeats per stimulus

% Set model parameters
guessRate   = 0.000;                % The fraction of guesses
stimSens    = .5;                   % Stimulus sensitvity parameter, higher values produce a steeper psychometric function, strictly positive
stimCrit    = 0;                    % The sensory decision criterion in units of stimulus magnitude (e.g., orientation in degrees)
uncMeta     = .5;                   % Meta-uncertainty: the second stage noise parameter, only affects confidence judgments, strictly positive
confCrit    = [.75 1];              % The confidence criteria, unitless (can include more than 1)
asymFlag    = 0;                    % If set to 1, it allows for asymmetrical confidence criteria and confCrit needs two times as many elements    

%create metauncertainty and confidence criterion arrays
metaUncArray = zeros(1,100);
confCritArray = linspace(0,5,100);

%generate (uncMeta,confCrit) pairs
%generate 100 values for each variable
countI = 1;
countJ = 1;
negLLArray = zeros(1,10000);
for i=linspace(0.01,5,100)
    uncMeta = log10(i);
    metaUncArray(countI) = uncMeta;
    for j=linspace(0,5,100)
        confCrit = j;
        modelParams = [guessRate, stimSens, stimCrit, uncMeta, confCrit];
        modelParamsLabel = [{'guessRate', 'stimSens', 'stimCrit', 'uncMeta'} repmat({'confCrit'},1,numel(confCrit))];
        
        % Set calulation precision
        calcPrecision = 100;                % Higher values produce slower, more precise estimates. Precision saturates after ~25
        
        % Get model predictions
        [choiceLlh] = getLlhChoice(stimValue, modelParams, calcPrecision, asymFlag);
        %disp(choiceLlh);
        % Simulate choice data
        randNumbers = rand(stimReps, numel(stimValue));
        criteria    = cumsum(choiceLlh);
        
        for iX = 1:size(criteria, 1)
            if iX == 1
                n{iX} = sum(randNumbers <= criteria(1,:));
            elseif (iX > 1 && iX < size(criteria, 1))
                n{iX} = sum((randNumbers > criteria(iX-1,:)) & (randNumbers <= criteria(iX,:)));
            elseif iX == size(criteria, 1)
                n{iX} = sum(randNumbers > criteria(end-1,:));
            end
        end
        nChoice  = cell2mat(n');
        %disp(nChoice);
        % Fit simulated data
        options  = optimset('Display', 'off', 'Maxiter', 10^5, 'MaxFuneval', 10^5);
        obFun    = @(paramVec) giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag);
        startVec = [.01 1 -0.1 0.5 sort(2*rand(1,numel(confCrit)))];
        % Search bounds:
        LB          = zeros(numel(startVec),1);
        UB          = zeros(numel(startVec),1);
        
        LB(1,1)     = 0;                        UB(1,1)        = 0.1;                   % Guess rate
        LB(2,1)     = 0;                        UB(2,1)        = 10;                    % Stimulus sensitivity
        LB(3,1)     = -3;                       UB(3,1)        = 3;                     % Stimulus criterion
        LB(4,1)     = 0.01;                     UB(4,1)        = 5;                     % Meta uncertainty 
        LB(5:end,1) = 0;                        UB(5:end,1)    = 5;                     % Confidence criteria
        [paramEst,NLL]    = fmincon(obFun, startVec, [], [], [], [], LB, UB, [], options);
        negLLArray(countJ) = NLL;
        countJ = countJ + 1;
    end
    countI = countI + 1;
end

%plot surface plot of negLLArray versus (metaUncArray,confCritArray) values
%convert negLLArray to a matrix (for plotting purposes)
Z = reshape(negLLArray,100,100);
surf(metaUncArray,confCritArray,Z);
title("NLL vs (Meta-Uncertainty, Confidence Criterion)");
xlabel("log(Meta-Uncertainty)"), ylabel("Confidence Criterion"), zlabel("NLL");


function [NLL] = giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag)
choiceLlh = getLlhChoice(stimValue, paramVec,calcPrecision, asymFlag);
NLL       = -sum(sum(nChoice.*log(choiceLlh)));
end