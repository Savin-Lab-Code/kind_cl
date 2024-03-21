%generates the wait time ratio and regression over training figure

%TODO: decide if Astructs_final or Astructs
%how to label each point


%load paths
addpath(genpath('/Users/dhocker/projects/constantinoplelab/Analysis/'));
addpath(genpath('/Users/dhocker/projects/dynamics/'));


%% run script


% number of sessions to use. split in half. 
% looks at first n/2 sessions, and last n/2 sessions

nsess = 50; 
grp = 2;

[B1, B2,P1, P2,dwt1, dwt2, pcatch1, pcatch2] =...
    macro_regress_overtraining_V4(nsess,'rat',grp);

%% other script for early stages of training. FOr RNNs
%just guessing on sessions. need to look
nsess = 100;
grp = 10;
[B1_catch, P1_catch] = macro_regress_earlystages_rnn(nsess, 'catch', Sdir)
[B1_nocatch, P1_nocatch] = macro_regress_earlystages_rnn(nsess, 'nocatch', Sdir)

sens_catch = B1_catch{1};
sense_nocatch = B1_nocatch{1};


%%

%use this as a proxy for now for linear sensitivity. will need a separate
%regression though
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

%nrat = size(wtratio_first,1); %TODO: I think this is inflated
nrat = sum(~isnan(wtratio_first(:,1))); %ignores omitted data

%% plot linear sensitivity and 

figure(44)
clf


tvec1 = 1:nsess/2;
buff = 10; %number of sessions for visual indent from first to last sess parts
tvec2 = nsess/2 + buff:nsess+buff-1;


t1 = [0,10,20]; %tick mark locations, early treaining
t2 = t1+sum(t1)+buff; %same, late training
ttxt = {};
tticks = [t1,t2];
for j = 1:numel(t1)
    ttxt{j} = num2str(t1(j)*grp);
end
t1r = fliplr(t1);
for j = 1:numel(t2)
    ttxt{j+numel(t1)} = num2str(t1r(j)*grp);
end



subplot(3,1,1)
hold on
shadedErrorBar(tvec1,nanmean(sensitivity_first), nanstd(sensitivity_first)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
shadedErrorBar(tvec2,nanmean(sensitivity_last), nanstd(sensitivity_last)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
plot(tvec1, nanmean(sensitivity_first),'linewidth',2, 'color','k')
plot(tvec2, nanmean(sensitivity_last),'linewidth',2, 'color','k')

%old way, likely wrong
xticks([0,10,20,buff+nsess/2,buff+nsess/2+10,buff+nsess])
xticklabels({'0',10*grp,20*grp,20*grp, 10*grp,'0'})

xticks(tticks)
xticklabels(ttxt)
%xticklabels({'0',num2str(nsess/2),num2str(nsess),num2str(nsess), num2str(nsess/2),'0'})
title(sprintf('Rat\n linear sensitivity'))


subplot(3,1,2)
hold on
shadedErrorBar(tvec1,nanmean(wtratio_first), nanstd(wtratio_first)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
shadedErrorBar(tvec2,nanmean(wtratio_last), nanstd(wtratio_last)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
plot(tvec1, nanmean(wtratio_first),'linewidth',2, 'color','k')
plot(tvec2, nanmean(wtratio_last),'linewidth',2, 'color','k')
xticks(tticks)
xticklabels(ttxt)

%xticks([0,10,20,40,50,60])
%xticklabels({'0',num2str(nsess/2),num2str(nsess),num2str(nsess), num2str(nsess/2),'0'})
title('wt ratio ')

subplot(3,1,3)
hold on
shadedErrorBar(tvec1,nanmean(blockcoeff_first), nanstd(blockcoeff_first)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
shadedErrorBar(tvec2,nanmean(blockcoeff_last), nanstd(blockcoeff_last)/sqrt(nrat),'lineprops',{'linewidth',1,'color','k'})
plot(tvec1, nanmean(blockcoeff_first),'linewidth',2, 'color','k')
plot(tvec2, nanmean(blockcoeff_last),'linewidth',2, 'color','k')
title('block coeff')


xlabel('<-- sessions from start                                   --> sessions from end)')
%xticks([0,10,20,40,50,60])
%xticklabels({'0',num2str(nsess/2),num2str(nsess),num2str(nsess), num2str(nsess/2),'0'})
xticks(tticks)
xticklabels(ttxt)
