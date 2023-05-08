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
stimValue = linspace(-4.5, 4.5, 19);   % The different stimulus conditions in units of stimulus magnitude (e.g., orientation in degrees)
stimReps  = 2000;                   % The number of repeats per stimulus

%MAIN GOAL: loop through different initial values of uncMeta and confCrit and generate
%LL surface for each
for metaUncVal = logspace(log10(0.1),log10(5),5)
    for confCritVal = linspace(0.25,5,5)
        % Set model parameters
        guessRate   = 0.000;                % The fraction of guesses
        %look at recent MTurk data and look at distribution of stimSens of
        %subjects, and try stimSens values (median, 25%, and 75%)
        %save figures for each stimSens value
        stimSens    = 0.1;                   % Stimulus sensitvity parameter, higher values produce a steeper psychometric function, strictly positive
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
        
        % number of samples meta uncertainty
        M = 100;
        % number of samples confidence criteria
        N = 100;
        
        %create metauncertainty and confidence criterion arrays
        metaUncArray = logspace(log10(0.1),log10(10),M);
        confCritArray = linspace(0.05,5,N);
        
        %generate (uncMeta,confCrit) pairs
        %generate 100 values for each variable
        LLArray = zeros(M,N);
        for i=1:M
            uncMeta = metaUncArray(i);
            for j=1:N
                confCrit = confCritArray(j);
                paramVec = [guessRate, stimSens, stimCrit, uncMeta, confCrit];
        
                % Fit simulated data
                %get LL instead for now
                LL   = -1 * giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag);
                LLArray(i,j) = LL;
                
            end
           
        end
        
        [maxLL,i] = max(LLArray,[],"all");
        [row,col] = ind2sub([M,N],i);
        maxMetaUncVal = metaUncArray(row);
        maxConfCritVal = confCritArray(col);
        fprintf('max meta %1.3f, max confidence %1.3f, max log likelihood %1.3f\n',maxMetaUncVal,maxConfCritVal,maxLL);
        %disp(maxMetaUncVal);
        %disp(maxConfCritVal);
        
        %create surface plot of negLLArray versus (metaUncArray,confCritArray) values
        [metaMesh,confMesh] = meshgrid(metaUncArray,confCritArray);
        % sPlot = surfc(metaUncArray,confCritArray,LLArray);
        LLPlot = LLArray';
        sPlot = surfc(metaMesh,confMesh,LLPlot);
        set(sPlot,'edgecolor','none');
        set(gca,'XScale','log');
        %export file with a descriptive name
        outputFileName = strcat("(", num2str(metaUncVal), ",", num2str(confCritVal), ")_LL_Tradeoff_Analysis.fig");
        title(strcat("(", num2str(metaUncVal), ",", num2str(confCritVal), ")", " Parameter Tradeoff Analysis"));
        xlabel("Meta-Uncertainty"), ylabel("Confidence Criterion"), zlabel("LL");
        hold on;
        
        %plot maximum LL value on surface plot
        scatter3(maxMetaUncVal,maxConfCritVal,maxLL,'r','filled');
        fprintf('meta unc val %1.3f, conf crit val %1.3f \n\n',metaUncVal,confCritVal);
        scatter3(metaUncVal,confCritVal,maxLL,'g','filled');
        hold off;

        %save figure
        fig = gcf;
        saveas(fig,outputFileName);
    end
end

 
function [NLL] = giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag)
choiceLlh = getLlhChoice(stimValue, paramVec,calcPrecision, asymFlag) + eps;
NLL       = -sum(sum(nChoice.*log(choiceLlh)));
end