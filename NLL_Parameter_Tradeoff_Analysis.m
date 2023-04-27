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

%MAIN GOAL: loop through different initial values of uncMeta and confCrit and generate
%NLL surface for each
for metaUncVal = linspace(0.01,5,5)
    for confCritVal = linspace(0,5,5)
        % Set model parameters
        guessRate   = 0.000;                % The fraction of guesses
        stimSens    = .5;                   % Stimulus sensitvity parameter, higher values produce a steeper psychometric function, strictly positive
        stimCrit    = 0;                    % The sensory decision criterion in units of stimulus magnitude (e.g., orientation in degrees)
        uncMeta     = metaUncVal;                   % Meta-uncertainty: the second stage noise parameter, only affects confidence judgments, strictly positive
        confCrit    = confCritVal;              % The confidence criteria, unitless (can include more than 1)
        asymFlag    = 0;                    % If set to 1, it allows for asymmetrical confidence criteria and confCrit needs two times as many elements   

        %generate single nChoice
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
        
        
        
        %create metauncertainty and confidence criterion arrays
        metaUncArray = logspace(log10(0.1),log10(10),100);
        confCritArray = linspace(0,5,100);
        
        %generate (uncMeta,confCrit) pairs
        %generate 100 values for each variable
        negLLArray = zeros(100,100);
        for i=1:100
            uncMeta = metaUncArray(i);
            for j=1:100
                confCrit = confCritArray(j);
                paramVec = [guessRate, stimSens, stimCrit, uncMeta, confCrit];
        
                % Fit simulated data
                NLL   = giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag);
                negLLArray(i,j) = NLL;
                
            end
           
        end
        
        [m,i] = min(negLLArray,[],"all");
        [row,col] = ind2sub([100,100],i);
        disp(metaUncArray(row));
        disp(confCritArray(col));
        
        %create surface plot of negLLArray versus (metaUncArray,confCritArray) values
        sPlot = surf(metaUncArray,confCritArray,negLLArray);
        %export file with a descriptive name
        outputFileName = strcat("(", num2str(metaUncVal), ",", num2str(confCritVal), ")_NLL_Tradeoff_Analysis.eps");
        title(strcat("(", num2str(metaUncVal), ",", num2str(confCritVal), ")", " Parameter Tradeoff Analysis"));
        xlabel("Meta-Uncertainty"), ylabel("Confidence Criterion"), zlabel("NLL");
        saveas(sPlot,outputFileName);
    end
end

 
function [NLL] = giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag)
choiceLlh = getLlhChoice(stimValue, paramVec,calcPrecision, asymFlag);
NLL       = -sum(sum(nChoice.*log(choiceLlh)));
end