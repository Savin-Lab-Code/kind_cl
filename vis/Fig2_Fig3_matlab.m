%Figure 1 matlab

addpath(genpath('~/projects/kind_cl'))
%addpath(genpath('~/projects/constantinoplelab/Analysis/RWT'))
%addpath(genpath('~/projects/constantinoplelab/Analysis/David/'))
%addpath(genpath('~/projects/constantinoplelab/Analysis/utils_core'))
%addpath(genpath('~/projects/dynamics'))


%% set some basic directories

datadir = '/Users/dhocker/projects/constantinoplelab/published/publisheddata_mah2023/';  
r = load(strcat(datadir,'ratList.mat'));
ratList = r.ratList;


%% load an process rat data

%[ratList, datapath, ~, a,~] = getmypaths('Final');
%sexList = a.sexList;


for rr = 1:length(ratList)
    fprintf('%d out of %d\n', rr, length(ratList));

    % Load Data
    try
        fname = strcat(['ratTrial_', ratList{rr}, '.mat']);
        a = load([datadir 'A_Structs_Final' filesep fname]);
        A = a.A;
    
        % Average wait time and SEM as a function of reward in each block
        [high, low, mix, ps(rr,1)] = wtcurves(A);
    
        WT.high(rr,:) = high.wt;
        WT.low(rr,:) = low.wt;
        WT.mixed(rr,:) = mix.wt;
    catch
        disp('skipping: ')
        disp(ratList{rr})
        nanvec = [nan, nan, nan, nan, nan];
        WT.high(rr,:) = nanvec;
        WT.low(rr,:) = nanvec;
        WT.high(rr,:) = nanvec;
    end

end

% Compare wait time for 20 uL in high vs. low blocks
pwt = signrank(WT.high(:,3), WT.low(:,3));
wt_ratio = WT.high(:,3)./WT.low(:,3);

% make sure you know the number of rats
test = WT.high(:,3);
nrats = sum(~isnan(test));
disp('Nrats')
disp(nrats)

%% Fig 2c: get wait-time curve for a rat

for j = 1:nrats
    %if strcmp(ratList{j},'C027')
    if strcmp(ratList{j},'S028')
        disp('i found you')
        fname = strcat(['ratTrial_', ratList{j}, '.mat']);
        a = load([datadir 'A_Structs_Final' filesep fname]);
        A = a.A;
        idx = j;
        break;
    end
end

figure(1) 
clf
plotblocks_together(A,ratList{idx}, 1, true)

%%
figure(1) 
clf

%goodlist = [2,4,16,22,23,26,33,38,42,50,63,66,67,68,73,75,81,102,115,135,144,147,149,155,157,158,182,187,202,220,223,231,232,245,249,257,273]

idx = 155

fname = strcat(['ratTrial_', ratList{idx}, '.mat'])
a = load([datadir 'A_Structs_Final' filesep fname]);
A = a.A;
%plotblocks_together(A,ratList{idx}, 1, true)
plotblocks_together(A,ratList{idx}, 1, true)







%% Fig 2d: distributions of wait time ratio

figure(14)
clf
hold on

wt_binwidth = 0.02;
alpha = 0.05;
% color scheme for figure
wt_color = '#a16ae8';
iti_color = '#50C878';

h1 = histogram(wt_ratio, facecolor='none', BinWidth=wt_binwidth);
histogram(wt_ratio(ps(:,1)<alpha),...
    facecolor=wt_color, BinEdges = h1.BinEdges)

xline(1, 'k--', linewidth=1, alpha=1)

xlim(1+[-0.27, 0.27])
ylim([0 50])
title(pwt)
xline(mean(wt_ratio,'omitnan'), 'r', linewidth=1, alpha=1)

xlabel('Wait time ratio (20 High/Low)')
ylabel('N (rats)')





%% Fig 2g: get probabilities of block from bayes-optimal model

probdir = strcat(datadir,'ModelFits/FitAll_WT_Bayes_24July23_Final/');

%probdir = '/Volumes/server/BehavioralModel/BayesSubOpt_OnlyLambda_flatModelFits/FitAll_WT_BayesSubOpt_24July23_Final/';

%probdir = '/Volumes/server/BehavioralModel/BayesSubOpt_OnlyLambda_nonflatModelFits/FitAll_WT_BayesSubOpt_nonflat_24July23_Final/';
f = load(strcat(probdir,'BestFit.mat'));
pdat = f.BestFit.S015.Test;

probs = pdat.Belief';
block = pdat.ratTrial.block;
block(block==1) = 0.5;
block(block==2) = 1;
block(block==3) = 0;
mask = ~isnan(pdat.ratTrial.wait_time);

%figure(18)
figure(19)
clf
hold on

plot(probs(mask,1),'color','k','linewidth',2)
plot(probs(mask,2),'color','r','linewidth',2)
plot(probs(mask,3),'color','b','linewidth',2)

plot(block(mask),'linewidth',0.5,'color','k','linestyle','--')
title('bayes optimal')




%% Fig2e: wait time dynamics for rats. TODO

aligntype = 'incongruent';
usecausal = true  % should be true
%[ltom, htom, mtol, mtoh] = block_wtdynamics_macro(20,10,'all','All',...
%                                                  aligntype, usecausal);


[ltom, htom, mtol, mtoh] = block_wtdynamics(20,10,'rat','rat',...
                                                  aligntype, usecausal);


%% Fig 3f: process the incongruent trial block dynamics
aligntype = 'incongruent';
usecausal = true
[ltom, htom, mtol, mtoh] = block_wtdynamics(20,10,'rnn','rnn', aligntype, usecausal);

%% Figure 2f and 3g: regression and WTR progress for rats or RNNs


%fig2do = 'rat'; % Fig. 2f
fig2do = 'rnn'; % Fig. 3g

switch fig2do
    case 'rat'
        nsess = 50; 
        grp = 2;
    case 'rnn'
        nsess = 100;
        grp = 2;
end

[B1, B2,P1, P2,dwt1, dwt2, pcatch1, pcatch2] =...
    macro_regress_overtraining(nsess,fig2do,grp);

%grab and filter data
sens_test_first = B1{1};
sens_test_last = B2{1};

sensitivity_first = B1{4};
sensitivity_last = B2{4};
%mask out insignificant slopes
sensitivity_first(P1 > 0.05) = nan; 
sensitivity_last(P2 > 0.05) = nan;

blockcoeff_first = B1{3};
blockcoeff_last = B2{3};

wtratio_first = dwt1;
wtratio_last = dwt2;

nrat = size(wtratio_first,1);

figure(44)
clf


tvec1 = 1:nsess/2;
buff = 10; %number of sessions for visual indent from first to last sess parts
tvec2 = nsess/2 + buff:nsess+buff-1;



subplot(2,1,1)
hold on
shadedErrorBar(tvec1,nanmean(wtratio_first), nanstd(wtratio_first)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
shadedErrorBar(tvec2,nanmean(wtratio_last), nanstd(wtratio_last)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
plot(tvec1, nanmean(wtratio_first),'linewidth',2, 'color','k')
plot(tvec2, nanmean(wtratio_last),'linewidth',2, 'color','k')

if strcmp(fig2do,'rat')
    xticks([0,10,20,40,50,60])
else
     xticks([0,25,50,60,85,110])
end
xticklabels({'0',num2str(nsess/2),num2str(nsess),num2str(nsess), num2str(nsess/2),'0'})
title('wt ratio ')

subplot(2,1,2)
hold on
shadedErrorBar(tvec1,nanmean(blockcoeff_first), nanstd(blockcoeff_first)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
shadedErrorBar(tvec2,nanmean(blockcoeff_last), nanstd(blockcoeff_last)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
plot(tvec1, nanmean(blockcoeff_first),'linewidth',2, 'color','k')
plot(tvec2, nanmean(blockcoeff_last),'linewidth',2, 'color','k')
title('block coeff')


xlabel('<-- sessions from start                                   --> sessions from end)')
if strcmp(fig2do,'rat')
    xticks([0,10,20,40,50,60])
else
     xticks([0,25,50,60,85,110])
end
xticklabels({'0',num2str(nsess/2),num2str(nsess),num2str(nsess), num2str(nsess/2),'0'})


