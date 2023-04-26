%%% Likelihood surface %%%%%
% Silvia 03.05.16 %

function pArray = likSurface(iterations,sData,v1,d1,v2,d2) 

% % addpath('/Users/silvia/Dropbox (Personal)/MATLAB/ITC/Analysis/group/simulations');
% load 'choiceSet.mat'
% v1 = choiceSet(:,1);
% d1 = zeros(length(choiceSet),1);
% v2 = choiceSet(:,2);
% d2 = choiceSet(:,3);
% load 'choiceSet2.mat'
% v1 = a(:,5);
% d1 = zeros(length(a),1);
% v2 = a(:,6);
% d2 = a(:,8);

param = nan(iterations,3);
for n = 1:iterations
    param(n,1) = (1.8-0.01)*rand + 0.1; %alpha
%     param(n,1) = (7.6-0.112)*rand + 0.112; %alpha
    param(n,2) = (0.2-0.0001)*rand + 0.0001; %kappa
%     param(n,2) = (0.4-0.0001)*rand + 0.0001; %kappa
    param(n,3) = (6-(0.001))*rand + (0.001); %beta
    
end
logLikelihoodSurface = zeros(length(param(:,1)),length(param(:,2)),length(param(:,3)));
for i=1:length(param(:,1))
    for j=1:length(param(:,2))
        for k=1:length(param(:,3))
        alpha = param(i,1);
        kappa = param(j,2);
        beta = param(k,3);
        [info] = local_negLL(sData,v1,d1,v2,d2,alpha,kappa,beta);
        logLikelihoodSurface(i,j,k) = info.LL;
        end
    end
end

pArray = cell(1,4);
pArray{1} = param(:,1);
pArray{2} = param(:,2);
pArray{3} = param(:,3);
pArray{4} = logLikelihoodSurface;

% [X,I] = sort(param(:,1)); %alpha
[X,I] = sort(param(:,3)); %beta
[Y,J] = sort(log(param(:,2))); %kappa

m = randi(50,1);

figure(1)
% surf(X,Y,logLikelihoodSurface(I,J,m)); %for alpha versus kappa
surf(X,Y,squeeze(logLikelihoodSurface(m,J,I))); %for kappa versus beta
% xlabel('Risk Parameter', 'FontSize', 16, 'FontWeight','Bold');
xlabel('Noise Parameter', 'FontSize', 16, 'FontWeight','Bold');
ylabel('ln(Discount Rate)', 'FontSize', 16, 'FontWeight','Bold');
zlabel('Negative Log Likelihood', 'FontSize', 16, 'FontWeight','Bold');
title('Likelihood Surface', 'FontSize', 20, 'FontWeight','Bold');

figure(2)
% meshc(X,Y,logLikelihoodSurface(I,J,m)); %for alpha versus kappa
meshc(X,Y,squeeze(logLikelihoodSurface(m,J,I))); %for kappa versus beta
% xlabel('Risk Parameter', 'FontSize', 16, 'FontWeight','Bold');
xlabel('Noise Parameter', 'FontSize', 16, 'FontWeight','Bold');
ylabel('ln(Discount Rate)', 'FontSize', 16, 'FontWeight','Bold');
zlabel('Negative Log Likelihood', 'FontSize', 16, 'FontWeight','Bold');
title('Likelihood Surface', 'FontSize', 20, 'FontWeight','Bold');
end

%----- LOG-LIKELIHOOD FUNCTION
function [info] = local_negLL(choice,v1,d1,v2,d2,alpha,kappa,beta)

p = choice_prob(v1,d1,v2,d2,alpha,kappa,beta);

% Trap log(0)
ind = p == 1;
p(ind) = 0.9999;
ind = p == 0;
p(ind) = 0.0001;
% Log-likelihood
err = (choice==1).*log(p) + (1 - (choice==1)).*log(1-p);
% Sum of -log-likelihood
sumerr = sum(err);

info.LL = sumerr;
end

%----- DISCOUNT FUNCTION - HYPERBOLIC
%     y = discount(v,d,kappa,alpha)
%
%     INPUTS
%     v     - values
%     d     - delays
%     kappa - discount rate
%     risk  - risk attitude parameter
%
%     OUTPUTS
%     y     - discounted values
%
%     REVISION HISTORY:
%     brian lau 03.14.06 written
%     khoi 06.26.09 simplified
%     silvia 03.05.16 added alpha

function y = discount(v,d,kappa,alpha)
alpha = 1;
y = (v.^alpha)./(1+kappa*d);
end

%----- CHOICE PROBABILITY FUNCTION - LOGIT
%     p = choice_prob(v1,d1,v2,d2,beta);
%
%     INPUTS
%     v1    - values of option 1 (ie, sooner option)
%     d1    - delays of option 1
%     v2    - values of option 2 (ie, later option)
%     d2    - delays of option 2
%     beta  - parameters, noise term (1), discount rate (2) and risk (3)
%
%     OUTPUTS
%     p     - choice probabilities for the **OPTION 2**
%
%     REVISION HISTORY:
%     brian lau 03.14.06 written
%     khoi 06.26.09 simplified
%     Silvia 03.05.16 included free alpha

function p = choice_prob(v1,d1,v2,d2,alpha,kappa,beta)

u1 = discount(v1,d1,kappa,alpha);
u2 = discount(v2,d2,kappa,alpha);

% logit, smaller beta = larger error
p = 1 ./ (1 + exp(beta*(u1-u2)));
end

