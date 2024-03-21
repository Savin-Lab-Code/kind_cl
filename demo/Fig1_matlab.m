%Figure 1 matlab


addpath(genpath('~/projects/constantinoplelab/Analysis/RWT'))
addpath(genpath('~/projects/constantinoplelab/Analysis/David/'))
addpath(genpath('~/projects/constantinoplelab/Analysis/utils_core'))
addpath(genpath('~/projects/dynamics'))

%% set some basic directories

datadir = '/Volumes/server/ProcessedRatData/';  


%% load an process rat data

[ratList, datapath, ~, a,~] = getmypaths('Final');
sexList = a.sexList;


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

%% get wait-time curve for a rat



%% distributions of wait time ratio

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





%% get probabilities of block from bayes-optimal model

probdir = '/Volumes/server/BehavioralModel/BayesModelFits/FitAll_WT_Bayes_24July23_Final/';

%probdir = '/Volumes/server/BehavioralModel/BayesSubOpt_OnlyLambda_flatModelFits/FitAll_WT_BayesSubOpt_24July23_Final/';

%probdir = '/Volumes/server/BehavioralModel/BayesSubOpt_OnlyLambda_nonflatModelFits/FitAll_WT_BayesSubOpt_nonflat_24July23_Final/';
%f = load(strcat(probdir,'BestFit.mat'))
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



%% wait time dynamics for rats

aligntype = 'incongruent';
usecausal = true  % should be true
[ltom, htom, mtol, mtoh] = block_wtdynamics_macro(20,10,'all','All',...
                                                  aligntype, usecausal);

%% wait time dynamoics for RNNS

reformat = false;

if reformat

    % preprocess the RNN files in matlab. they might not have .catch, or A
    % struct as expected in the behavioral data
    
    loaddir = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/mat/';
    
    %loaddir = '/Users/dhocker/projects/dynamics/results/20230206_clstudy/nok_cl/mat/';
    
    f = dir(strcat(loaddir,'*.mat'))
    for j = 1:numel(f)
    %for j = 1:1
        fname = f(j).name
        A = load(strcat(loaddir,fname));
        if isfield(A,'A')
            continue
        else
            if isfield(A,'catch_')
                A.catch = A.catch_;
                A = rmfield(A,'catch_');
            end
            save(strcat(loaddir,fname),'A')
        end
    end
end


%% process the incongruent trial block dynamics
aligntype = 'incongruent';
usecausal = true
[ltom, htom, mtol, mtoh] = block_wtdynamics_macro(20,10,'all','RNN_full_cl', aligntype, usecausal);
%%

aligntype = 'block';
usecausal = false
[ltom, htom, mtol, mtoh] = block_wtdynamics_macro(10,10,'all','RNN_nok_cl', aligntype, usecausal);

%%
aligntype = 'block';
usecausal = false
[ltom, htom, mtol, mtoh] = block_wtdynamics_macro(10,10,'all','RNN_ra', aligntype, usecausal);



%% looking for reverse adapters
%expt = 'all'
%[ratList, datapath, ~] = getmypaths(expt);

wt_ratio_nan = wt_ratio;
wt_ratio_nan(isnan(wt_ratio)) = -1;
[~,sortidx] = sort(wt_ratio_nan,'descend');



idx = 5;

disp(ratList{sortidx(idx)})
fname = strcat(['ratTrial_', ratList{sortidx(idx)}, '.mat']);
disp(fname)
load(strcat([datapath, fname]));
    
[~, A.reward] = convertreward(A.reward);

figure(1)
clf

plotblocks_together(A,ratList{sortidx(idx)}, 1, true)

%% reverse adaptive rats
aligntype = 'block';
usecausal = true
[ltom, htom, mtol, mtoh] = block_wtdynamics_macro(10,10,'all','RA_rats', aligntype, usecausal);
