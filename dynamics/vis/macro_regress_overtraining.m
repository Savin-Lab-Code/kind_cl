function [B1, B2,P1, P2,dwt1, dwt2,pcatch1, pcatch2] =...
    macro_regress_overtraining(nsess, dattype, grp, Sdir)

%modified version of Christine's "V2" code, 
%in Analysis/christine/behavior/training
%includes regression of just current wait time
%to get linear sensitivity
%
%INPUTS:
%   datype: (str) 'rat' or 'rnn'. hardcoded path for RNN, for debugging.
%   grp: (int) number of groups


if strcmp(dattype,'rat')
    % paths to zenodo data from Mah et al. for rat data
    datapath = '/Users/dhocker/projects/constantinoplelab/published/publisheddata_mah2023/A_Structs_Final/';  
    datapath_0 = '/Users/dhocker/projects/constantinoplelab/published/publisheddata_mah2023/';  
    r = load(strcat(datapath_0,'ratList.mat'));
    ratList = r.ratList;
else
    % path to zenodo data from hocker et al. for rnn data
    datapath = '/Users/dhocker/projects/kind_cl/data/rnndata_matlab/overtraining/';
    %ratList = {'1','2','3','4','5','6','7','8','9',...
    %           '10','11','12','13','14','15','16','17','18','19','20'};
    ratList = {'1','2','3','4','5','6','7','8','9',...
               '10','11','12','13','14','15','16','17','18','19','20',...
               '21','22','23','24','25','26','27','28','29','30',...
               '31','32','33','34','35','36','37','38','39','40',...
               '41','42','43','44','45','46','47','48','49','50'};

end

[B1, B2, B1iti, B2iti] = deal(cell(1, 4));
[P1, P2] = deal(nan(length(ratList), ceil(nsess/grp)));

for k = 1:4
    B1{k} = nan(length(ratList), ceil(nsess/grp));
    B2{k} = nan(length(ratList), ceil(nsess/grp));
    B1iti{k} = nan(length(ratList), ceil(nsess/grp));
    B2iti{k} = nan(length(ratList), ceil(nsess/grp));
end

dwt1 = nan(length(ratList), ceil(nsess/grp)); dwt2 = dwt1;
diti1 = dwt1; diti2 = dwt1;
pcatch1 = dwt1; pcatch2 = dwt1;
pvio_iti1 = dwt1;
pvio_iti2 = dwt2;


for rr = 1:length(ratList)
    disp(rr);
    
    % Load Data
    if strcmp(dattype,'rat')
        fname = strcat(['ratTrial_', ratList{rr}, '.mat']);
    else
        fname = strcat(['rnn_',ratList{rr},'_allbeh.mat']);
        disp(fname)
    end
    
    if ~isfile(strcat([datapath,fname]))
        disp('skipping')
        continue
    end
        
    try
        if strcmp(dattype,'rat')
            %load(strcat([Sdir,filesep,ratList{rr},filesep,fname]));
            load(strcat([datapath,fname]));
            
        else
            %A = parse_rnndata4matlab(strcat([datapath,fname]));
            load(strcat([datapath,fname]));
            A.block = double(A.block);
        end
        A.wait_time(A.wait_time>A.wait_thresh) = nan; %TODO: A 
        A.ITI(A.ITI>prctile(A.ITI,99)) = nan;
        ctr = 1;
        
        firstn = 1:nsess;
        lastn = length(A.ntrials)-nsess+1:length(A.ntrials);
        
        tctr1 = 1;
        tctr2 = sum(A.ntrials(1:length(A.ntrials)-nsess))+1;
        for k = 1:grp:nsess
            
            these = firstn(k):firstn(k)+grp-1;
            if sum(these>length(A.ntrials))>0 %if indices surpass data, nan it.
                beta1 = nan(3,1);
                beta1iti = nan(3,1);
                dwt1(rr,ctr) = nan;
                diti1(rr,ctr) = nan;
                pcatch1(rr,ctr) = nan;
                pvio_iti1(rr,ctr) = nan;
                b_sensitivity1 = nan;
                pwait_1 = nan;
            else
                idx = tctr1:tctr1+sum(A.ntrials(these))-1;
                [A1] = parseSstruct(A, idx, 0);
                
                tctr1 = tctr1+sum(A.ntrials(these));
                %doITI = false;
                [beta1, beta1iti, dwt1(rr,ctr), diti1(rr,ctr)] =...
                    regress_trials_blocksv2(A1, 1);
                pcatch1(rr,ctr) =...
                    mean(A1.optout, 'omitnan')/mean(A1.prob_catch, 'omitnan');
                
                %do normal regression of linear sensitivity
                %regress_wt_vs_rew(A, nback, plotarg, cond)
                nback = 0;
                [b_sensitivity1, ~,pwait_1] = regress_wt_vs_rew(A1, nback);
                
                %ITI regression
                %[A1] = parseSstruct(A, idx, 1);
                %[~, ~, ~, ~, pvio_iti1(rr,ctr)] =...
                %    regress_trials_blocksv2(A1, 1, doITI);
                pvio_iti1(rr,ctr) = nan;
            end
            
            these = lastn(k):lastn(k)+grp-1;
            if sum(these<=0)>0 %if there are negative indices, nan it.
                beta2 = nan(3,1);
                beta2iti = nan(3,1);
                dwt2(rr,ctr) = nan;
                diti2(rr,ctr) = nan;
                pcatch2(rr,ctr) = nan;
                pvio_iti2(rr,ctr) = nan;
                b_sensitivity2 = nan;
                pwait_2 = nan;
            else
                %dont' use violation trials
                idx = tctr2:tctr2+sum(A.ntrials(these))-1;
                [A2] = parseSstruct(A, idx, 0);
                
                tctr2 = tctr2+sum(A.ntrials(these));
                %regress
                [beta2, beta2iti, dwt2(rr,ctr), diti2(rr,ctr)] =...
                    regress_trials_blocksv2(A2, 1);
                pcatch2(rr,ctr) =...
                    mean(A2.optout, 'omitnan')/mean(A2.prob_catch, 'omitnan');
                
                %do normal regression of linear sensitivity
                %[b_sensitivity2, ~, pwait_2] = regress_trials_currentrew(A2);
                [b_sensitivity2, ~, pwait_2] = regress_wt_vs_rew(A2, nback);
                

                %regress using optouts for ITI
                %[A2] = parseSstruct(A, idx, 1);
                %[~, ~, ~, ~, pvio_iti2(rr,ctr)] =...
                %    regress_trials_blocksv2(A2, 1,doITI);
                pvio_iti2(rr,ctr) = nan;
                
            end
            for jj = 1:3
                B1{jj}(rr,ctr) = beta1(jj);
                B2{jj}(rr,ctr) = beta2(jj);
                
                B1iti{jj}(rr,ctr) = beta1iti(jj);
                B2iti{jj}(rr,ctr) = beta2iti(jj);
            end
            B1{4}(rr,ctr) = b_sensitivity1;
            B2{4}(rr,ctr) = b_sensitivity2;
            P1(rr,ctr) = pwait_1;
            P2(rr,ctr) = pwait_2;
            
            ctr = ctr+1;
        end
        
        dwt1(rr,:) = smooth(dwt1(rr,:));
        dwt2(rr,:) = smooth(dwt2(rr,:));
        
        diti1(rr,:) = smooth(diti1(rr,:));
        diti2(rr,:) = smooth(diti2(rr,:));
        
        B1{jj}(rr,:) = smooth(B1{jj}(rr,:));
        B2{jj}(rr,:) = smooth(B2{jj}(rr,:));
        
        B1iti{jj}(rr,:) = smooth(B1iti{jj}(rr,:));
        B2iti{jj}(rr,:) = smooth(B2iti{jj}(rr,:));
        
        pcatch1(rr,:) = smooth(pcatch1(rr,:));
        pcatch2(rr,:) = smooth(pcatch2(rr,:));
        
        pvio_iti1(rr,:) = smooth(pvio_iti1(rr,:));
        pvio_iti2(rr,:) = smooth(pvio_iti2(rr,:));
    catch ME
        rethrow(ME)
    end
end

dwt1 = rmvoutlier(dwt1);
dwt2 = rmvoutlier(dwt2);
diti1 = rmvoutlier(diti1);
diti2 = rmvoutlier(diti2);
pcatch1 = rmvoutlier(pcatch1);
pcatch2 = rmvoutlier(pcatch2);
pvio_iti1 = rmvoutlier(pvio_iti1);
pvio_iti2 = rmvoutlier(pvio_iti2);
end

function [A1] = parseSstruct(A, ix, usepvio)

A1.block = A.block(ix);
A1.wait_time = A.wait_time(ix);
A1.reward = A.reward(ix);
A1.vios = A.vios(ix);
A1.optout = A.optout(ix);
A1.hits = A.hits(ix);
A1.ITI = A.ITI(ix);
A1.prob_catch = A.prob_catch(ix);
A1.catch = A.catch(ix);

if usepvio==0
    bad = find(A1.vios==1);
    A1.block(bad) = [];
    A1.wait_time(bad) = [];
    A1.reward(bad) = [];
    A1.vios(bad) = [];
    A1.optout(bad) = [];
    A1.hits(bad) = [];
    A1.ITI(bad) = [];
    A1.prob_catch(bad) = [];
    A1.catch(bad) = [];
elseif usepvio==1
    bad = find(A1.vios~=1);
    bad = bad+1; bad(bad>length(A1.vios)) = [];
    A1.block(bad) = [];
    A1.wait_time(bad) = [];
    A1.reward(bad) = [];
    A1.vios(bad) = [];
    A1.optout(bad) = [];
    A1.hits(bad) = [];
    A1.ITI(bad) = [];
    A1.prob_catch(bad) = [];
    A1.catch(bad) = [];
end
end

function [x] = rmvoutlier(x)
x(isoutlier(x, 'movmedian', 5)) = nan;
end
