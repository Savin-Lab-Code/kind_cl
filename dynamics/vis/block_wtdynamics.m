function [ltom, htom, mtol, mtoh] = block_wtdynamics(twin,...
    smoothfactor, expt, typearg,varargin)
  %%plots average change in wait time around block dynamics. cmc 10/19/21.
  % modified dlh 2024
  %inputs: twin = time window, or number of trials around block transition. 
    %      smoothfactor for n-trial smoothing. 7-10 is good.
    %      typearg specifies which dataset ('rat' or 'rnn')
    %      varargin contains alignment type and if causal smoothing is used

if numel(varargin) > 0
    aligntype = varargin{1}; % alignment type. 'block' or 'incongruent'
else
    aligntype = 'block';
end

if numel(varargin) > 1
    usecausal = varargin{2};
else
    usecausal = false;
end


switch typearg
    case 'rat'
        % paths to zenodo data from Mah et al. for rat data
        datapath = '/Users/dhocker/projects/constantinoplelab/published/publisheddata_mah2023/A_Structs_Final/';  
        datapath_0 = '/Users/dhocker/projects/constantinoplelab/published/publisheddata_mah2023/';  
        r = load(strcat(datapath_0,'ratList.mat'));
        ratList = r.ratList;
        titlevec = 'rat';
    case 'rnn'
        % path to zenodo data from hocker et al. for rnn data
        datapath = '/Users/dhocker/projects/kind_cl/data/rnndata_matlab/';
        names = dir(strcat(datapath,'*.mat'));
        ratList = cell(numel(names),1);
        for j = 1:numel(names)
            nj = split(names(j).name,'.');
            njj = split(nj{1},'ratTrial_');
            ratList{j} = njj{end};
        end
        titlevec = 'full cl';
    otherwise
        disp('incorrect typearg supplied. must be "rat" or "rnn" ')
end


ltom = nan(length(ratList), twin*2+1);
htom = ltom;
mtol = ltom;
mtoh = ltom;
xvec = -twin:1:twin;
for k = 1:length(ratList)
    
    disp(k)
    
    fname = strcat(['ratTrial_', ratList{k}, '.mat']);
    disp(strcat([datapath,fname]))
    load(strcat([datapath, fname]));
    
    if strcmp(aligntype, 'block')
        [ltom(k,:), htom(k,:), mtol(k,:), mtoh(k,:)] = block_dynamics_wt(A, twin, smoothfactor, true(size(A.wait_time)), usecausal);
    elseif strcmp(aligntype, 'incongruent')
        [~,~,mtol(k,:), mtoh(k,:),ltom(k,:), htom(k,:), ] = block_dynamics_wt(A, twin, smoothfactor, true(size(A.wait_time)), usecausal);
    else
        disp('incorrect alignment type. choose "block" or "incongruent" ')
    end
end


figure; 
subplot(1,2,1);
shadedErrorBar(xvec, mean(ltom, 'omitnan'), ...
    std(ltom, 'omitnan')./sqrt(length(ltom(:,1))), 'lineprops', '-b');
shadedErrorBar(xvec, mean(htom, 'omitnan'), ...
    std(htom, 'omitnan')./sqrt(length(htom(:,1))), 'lineprops', '-r');
hold on
set(gca, 'TickDir', 'out'); box off
switch aligntype
    case 'block'
        xlabel('Trial from block switch');
    case 'incongruent'
        xlabel('Trials from incongruent trial');
end
yl = ylim;
line([0 0], [yl(1) yl(2)], 'Color', [0 0 0], 'LineStyle', '--');
ylabel('\Delta z-scored wait time');
title('Adaptation into mixed');

subplot(1,2,2);
shadedErrorBar(xvec, mean(mtoh, 'omitnan'), ...
    std(mtoh, 'omitnan')./sqrt(length(mtoh(:,1))), 'lineprops', '-r');
shadedErrorBar(xvec, mean(mtol, 'omitnan'), ...
    std(mtol, 'omitnan')./sqrt(length(mtol(:,1))), 'lineprops', '-b');
hold on
set(gca, 'TickDir', 'out'); box off
xlabel('Trial from block switch');

yl = ylim;
line([0 0], [yl(1) yl(2)], 'Color', [0 0 0], 'LineStyle', '--');
ylabel('\Delta z-scored wait time');
title('Mixed into adaptation');

set(gcf, 'Color', [1 1 1]);
set(gcf, 'Position', [530 525 1151 420])
sgtitle(titlevec)