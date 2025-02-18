function [ltom, htom, mtol, mtoh] = block_wtdynamics_macro(twin,...
    smoothfactor, expt, typearg,varargin)
  %%plots average change in wait time around block dynamics. cmc 10/19/21.
  %inputs: twin = time window, or number of trials around each block transition. 
    %      smoothfactor is smoothing factor, because data is noisy. 7-10 is good.
    %      sexarg specifies which sex to use ('All', 'M', or 'F').
    %      expt is 'estrus_up' or 'estrus_down'

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

%{

switch sexarg
    case 'RNN_full_cl'
        disp('skipping getting rat list, rnn')
    case 'RNN_nok_cl'
        disp('skipping getting rat list, rnn')
    case 'RNN_ra'
        disp('skipping getting rat list, rnn')
    case 'RNN_pkind_mem'
        disp('skipping getting rat list, rnn')
    case 'RNN_pkind_mem_adapters'
        disp('skipping getting rat list, rnn')
    otherwise
        [ratList_all, datapath, ~, a,~] = getmypaths(expt);
        sexList = a.sexList;
end

if strcmpi(sexarg, 'All')
    
    ratList = ratList_all;
    titlevec = 'all';
elseif strcmpi(sexarg, 'M')
    ratList = ratList_all(strcmp(sexList, 'M'));
    titlevec = 'M';
elseif strcmpi(sexarg, 'F')
    ratList = ratList_all(strcmp(sexList, 'F'));
    titlevec = 'F';
elseif strcmpi(sexarg, 'RNN_full_cl')
    datapath = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/mat/'
    names = dir(strcat(datapath,'*.mat'))
    for j = 1:numel(names)
        nj = split(names(j).name,'.')
        njj = split(nj{1},'ratTrial_')
        ratList{j} = njj{end}
    end
    titlevec = 'full cl';
elseif strcmpi(sexarg, 'RNN_full_cl_nocatch')
    datapath = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/mat/nocatch/'
    names = dir(strcat(datapath,'*.mat'))
    for j = 1:numel(names)
        nj = split(names(j).name,'.')
        njj = split(nj{1},'ratTrial_')
        ratList{j} = njj{end}
    end
    titlevec = 'full cl';
elseif strcmpi(sexarg, 'RNN_nok_cl')
    datapath = '/Users/dhocker/projects/dynamics/results/20230206_clstudy/nok_cl/mat/'
    names = dir(strcat(datapath,'*.mat'))
    for j = 1:numel(names)
        nj = split(names(j).name,'.')
        njj = split(nj{1},'ratTrial_')
        ratList{j} = njj{end}
    end
    titlevec = 'classic cl';
elseif strcmpi(sexarg, 'RNN_ra')
    datapath = '/Users/dhocker/projects/dynamics/results/20230206_clstudy/nok_cl/mat/'
    names = dir(strcat(datapath,'ratTrial*.mat'))
    for j = 1:numel(names)
        nj = split(names(j).name,'.')
        njj = split(nj{1},'ratTrial_')
        ratList{j} = njj{end}
    end
    g = load(strcat(datapath,'wtr_rnn.mat'))
    ratList = g.ratList(g.wtr > 1.0)
    titlevec = 'reverse adapters from classic CL';
elseif strcmpi(sexarg, 'RA_rats')
    g = load('/Users/dhocker/projects/dynamics/results/rats_wtr.mat')

    ratList = ratList_all(g.wtr>1.0);
    titlevec = 'reverse adapting rats';
elseif strcmpi(sexarg, 'RNN_pkind_mem')
    datapath = '/Users/dhocker/projects/dynamics/results/20231003/pkind_mem/mat/'
    names = dir(strcat(datapath,'*.mat'))
    for j = 1:numel(names)
        nj = split(names(j).name,'.')
        njj = split(nj{1},'ratTrial_')
        ratList{j} = njj{end}
    end
    titlevec = 'pkind mem';
elseif strcmpi(sexarg, 'RNN_pkind_mem_adapters')
    disp('in here')
    datapath = '/Users/dhocker/projects/dynamics/results/20231003/pkind_mem/mat/adapters/'
    names = dir(strcat(datapath,'*.mat'))
    for j = 1:numel(names)
        nj = split(names(j).name,'.')
        njj = split(nj{1},'ratTrial_')
        ratList{j} = njj{end}
    end
    titlevec = 'pkind mem adapters';
else
    disp('incorrect sexarg')
end

%}

% TODO: remove all stuff above

switch typearg
    case 'rat'

        datadir = '/Users/dhocker/projects/constantinoplelab/published/publisheddata_mah2023/';  
        r = load(strcat(datadir,'ratList.mat'));
        ratList = r.ratList;
        titlevec = 'rat';
    case 'rnn'
        datapath = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/mat/';
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