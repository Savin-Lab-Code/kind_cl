%% study to plot the flow fields for different RNNs and different regions


set(0, 'DefaultFigureRenderer', 'painters');
%specify RNN, types, and options. these can be changed---------------------

%type of training?: 'full_cl'=kind CL in paper 'full_cl_test'=kind CL, 'nok_cl'= no CL
dytype = 'nok_cl'; 
num = 38;  %which RNN to select?
bgtype = 'gradient';  % background type. 'block', 'gradient', 'pi' for STR
res = [2,2,2]; %  resolution for quivers for each X,Y,Z dim
dbase = '/Users/dhocker/projects/dynamics/results/20231003/'; %data location

reg_idx = 0;  %which RNN region to select? OFC = 1, STR = 2


%options that should not really be changed. will break code----------------

isgen = false;
epoch = 'wait'; %epoch of trial
stage = 'block'; % stage of training
blocktype = 'mixed'; %block type to use if looking at STR
fnum = 17;  % the number associated with figure handle
zuse_idx = 1; %which z index to use. this data hard-coded to have only 1

%idx within stage of training
switch dytype
    case 'nok_cl'
        idx = 60;
    case 'nok_nocl'
        idx = 40;
    otherwise
        idx = 10;
end



%the data loading and plotting code----------------------------------------

%load data files
datadir = ''; %TODO
[f,g] = call(dbase, dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen);

%plot the flow fields

[h] = genflowfield(fnum, f,g,reg_idx, res, bgtype, zuse_idx);

% set title, font size
figure(fnum)
title(num2str(num))
set(gca,'fontsize',15)


%add the fixed points. creates second plot
%each fixed point has a been clustered, and given an assigned id using 
%g.labels

%mask = g.labels > -1;  %labels with -1 are considered noise. ignore
mask = g.labels > -2;  %should include noise

scatter(g.crds(mask,1), g.crds(mask,2),20, 'markeredgecolor','k')

figure(163)
clf
hold on
labs = sort(unique(g.labels));
for j = 1:numel(labs)
    if labs(j) > -1
        scatter(g.crds(g.labels==labs(j),1), g.crds(g.labels==labs(j),2),20,'markerfacecolor','black', 'markeredgecolor','black')
    elseif labs(j) ==-1  % a noise cluster
        scatter(g.crds(g.labels==labs(j),1), g.crds(g.labels==labs(j),2),20, 'markerfacecolor','red', 'markeredgecolor','red')
    end
end

%for each type of fixed point, calculate eigenvalues, plot schur modes
for k = 1:2
    switch k
        case 1
            figure(163)
        case 2
            figure(fnum)
    end
    pt_list = [1,257];

    for j = 1:numel(labs)
        if labs(j) ==-1
            continue
        end

        t = find(g.labels==labs(j));
        pt_idx = t(1);

        %use eigenvalue or schur deomposition? Schur, has orthogonal modes
        %[efuns,evals] = eig(squeeze(g.jac_pc(pt_idx,:,:)))
        [efuns,evals] = schur(squeeze(g.jac_pc(pt_idx,:,:)));

        % print eigenvalues to terminal. only do it on first loop
        if k == 1
            disp(strcat('[',num2str(evals(1,1)), ',', num2str(evals(2,2)),']'))
            disp('---')
        end

        %plot eigenvalues for each schur mode overlaid on flow filed
        sc1 = range(f.Y0)*0.2;
        sc2 = range(f.Y0)*0.2;

        plot([g.crds(pt_idx,1),g.crds(pt_idx,1)+sc1*real(efuns(1,1))],[g.crds(pt_idx,2),g.crds(pt_idx,2)+sc2*real(efuns(2,1))],'linewidth',1,'color','k')
        plot([g.crds(pt_idx,1),g.crds(pt_idx,1)+sc1*real(efuns(1,2))],[g.crds(pt_idx,2),g.crds(pt_idx,2)+sc2*real(efuns(2,2))],'linewidth',1,'color','k')

        text(g.crds(pt_idx,1)+1.2*sc1*real(efuns(1,1)), g.crds(pt_idx,2)+1.2*sc2*real(efuns(2,1)),(num2str(evals(1,1))), 'color','k','fontsize',15)
        text(g.crds(pt_idx,1)+1.2*sc1*real(efuns(1,2)), g.crds(pt_idx,2)+1.2*sc2*real(efuns(2,2)),(num2str(evals(2,2))), 'color','k','fontsize',15)

        if g.isfixed(pt_idx)
            text(g.crds(pt_idx,1)-0.2*sc1*real(efuns(1,2)), g.crds(pt_idx,2)-0.2*sc2*real(efuns(2,2)),'fp', 'color','k','fontsize',15)
        else
            text(g.crds(pt_idx,1)-0.2*sc1*real(efuns(1,2)), g.crds(pt_idx,2)-0.2*sc2*real(efuns(2,2)),'sp', 'color','k','fontsize',15)
        end
    end
end


% add sample trajectories from each block

figure(fnum); % return to main plot
colors = {'blue','black','red'};

switchvec = [3,1,2]; %f.blockp is orderd in mixed, high, low. this switches it

for j = 1:3
    disp(j)
    switch j
        case 1
            duse = f.samps_low;
        case 2
            duse = f.samps_mixed;
        case 3
            duse = f.samps_high;
    end
    
    % get best sample: ones with highest probability of being in each block
    ns = numel(duse);
    logpvals = nan(ns,3);
    zuse_m = zuse_idx;
    for k = 1:ns
        if size(duse{k}) > 1
             pc0 = duse{k}(2,1:3);
             [~,idx1] = min(abs(f.X0-pc0(1)));
             [~,idx2] = min(abs(f.Y0-pc0(2)));
             [~,idx3] = min(abs(f.Z0-pc0(3)));
             logpvals(k,:) = f.blockp(idx1,idx2,idx3,:);
        end
     end
     pvals = exp(logpvals)./sum(exp(logpvals),2);
     pvals(isnan(pvals)) = 0;

     [~,idx_use] = sort(pvals(:,switchvec(j)),'descend');

     
     nuse = 3;  % how many samples from each block to plot?
     for m = 1:nuse
         duse_m = duse{idx_use(m)};  % choose ones and identify their spot
         disp(strcat('sample for ',num2str(m)))
         disp(size(duse_m))  

        plot3(duse_m(2:end,1), duse_m(2:end,2),duse_m(2:end,3),'linewidth',2, 'color',colors{j})
        scatter3(duse_m(2,1), duse_m(2,2), duse_m(2,3),40, 'o',  'markerfacecolor',colors{j},'markeredgecolor',colors{j})
     end

end


%% function calls

function [f,g] = call(dbase, dytype,num,reg_idx,idx,epoch,stage,blocktype, isgen)

    fs = filesep; %file separator. mac/linux= '/'. windows = '\'

    % which set of files
    switch dytype
        case 'full_cl'
            %/Users/dhocker/projects/dynamics/results/20231003/

            dbase = strcat(dbase,'full_cl',fs);
            datadir = strcat(dbase,'dynamics',fs,'flows',fs);
            datadir_KE = strcat(dbase,fs,'dynamics',fs,'KEmin_constrained',fs);

        case 'full_cl_test'       
            dbase = strcat(dbase,'full_cl',fs);
            datadir = strcat(dbase,'dynamics',fs,'flows',fs,'flows_end',fs);
            datadir_KE = strcat(dbase,'dynamics',fs,'KEmin_constrained',fs,'2d',fs);
    
        case 'nok_cl'
            dbase = strcat(dbase,fs,'nok_cl',fs);
            datadir = strcat(dbase, fs,'dynamics',fs,'flows',fs);
            datadir_KE = strcat(dbase,fs,'dynamics',fs,'KEmin_constrained',fs,'2d',fs);
           
    end

    if reg_idx == 0
        if isgen
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_0_mixed_',epoch,'.mat');
        else
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_0_mixed_',epoch,'_boutique','.mat');
        end
    else
        if isgen
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_1_',blocktype,'_',epoch,'.mat');
        else
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_1_',blocktype,'_',epoch,'_boutique.mat');
        end
    end

    datname_KE = strcat(datadir_KE,'kemin_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_',num2str(reg_idx),'_',blocktype,'_',epoch,'.mat');
    
    f = load(datname);
    if isgen
        g = f;
    else
        disp(datname_KE)
        g = load(datname_KE);
    end

    %rename the samples, because there was a mistake
    %incorrect ordering: mixed, high, low, 5, 10, 20,40,80,...
    %correct ordering  :  5     10,   20, 40 ,80,  M, H, L
    %TODO: is this needed for the nok_cl data? I think this bug was fixed
    %by then....

    if ~strcmp(dytype, 'nok_cl')

        f.samps5new = f.samps_mixed;
        f.samps10new = f.samps_high;
        f.samps20new = f.samps_low;
        f.samps40new = f.samps_5;
        f.samps80new = f.samps_10;
        f.sampsMnew = f.samps_20;
        f.sampsHnew = f.samps_40;
        f.sampsLnew = f.samps_80;
    
        f.samps_5 = f.samps5new;
        f.samps_10 = f.samps10new;
        f.samps_20 = f.samps20new;
        f.samps_40 = f.samps40new;
        f.samps_80 = f.samps80new;
        f.samps_mixed = f.sampsMnew;
        f.samps_high = f.sampsHnew;
        f.samps_low = f.sampsLnew;
    
        f = rmfield(f,'samps5new');
        f = rmfield(f,'samps10new');
        f = rmfield(f,'samps20new');
        f = rmfield(f,'samps40new');
        f = rmfield(f,'samps80new');
        f = rmfield(f,'sampsMnew');
        f = rmfield(f,'sampsHnew');
        f = rmfield(f,'sampsLnew');
    end

end


function [h] = genflowfield(fnum, f,~, reg_idx, res, bgtype, zuse_idx)
    % will make the figure, then output the handle

    res1 = res(1);
    res2 = res(2);
    res3 = res(3);
    
    h = figure(fnum);
    clf
    hold on

    xvec = f.X0; %PC2. accounts for surf() flip convention
    yvec = f.Y0; % PC1. 
    zvec = f.Z0; % PC3
    
    %length of each dimension
    nx = length(xvec);
    ny = length(yvec);
    
    % loop over each z index and plot flow field in slices of z dim
    for m = 1:numel(zuse_idx)
        zuse_m = zuse_idx(m);
        [X,Y] = meshgrid(xvec,yvec);
    
        %decide on what to plot in background of flow field
        switch bgtype
            case 'block'  %probability of being in each block
                blocp = f.blockp;
                if reg_idx == 0
                    pblocks = zeros(nx,ny,3);
                    for j = 1:nx
                        for k = 1:ny
                            pblocks(j,k,:) = softmax(squeeze(blocp(j,k,1,:)));
                        end
                    end
                end
        
                %try nan-out unwanted bits: where block is not highest prob

                %mixed block
                maskmixed = 0*ones(size(pblocks(:,:,1)));
                maskmixed(pblocks(:,:,1) < pblocks(:,:,2) + pblocks(:,:,3)) = nan;
        
                %high block
                maskhigh = ones(size(pblocks(:,:,1)));
                maskhigh(pblocks(:,:,2) < pblocks(:,:,1) + pblocks(:,:,3)) = nan;
        
                %low block
                masklow = 2*ones(size(pblocks(:,:,2)));
                masklow(pblocks(:,:,3) < pblocks(:,:,1) + pblocks(:,:,2)) = nan;

                %combine all 3 masks to create a matrix of discrete values
                %that convey which block has highest prob at that point
                maskall = nan*ones(3,size(maskmixed,1),size(maskmixed,2));
                maskall(1,:,:) = maskmixed;
                maskall(2,:,:) = maskhigh;
                maskall(3,:,:) = masklow;
                maskall = squeeze(sum(maskall,1,'omitnan'));
        
                %plot
                shading interp
                hh = imagesc([xvec(1), xvec(end)], [yvec(1), yvec(end)],maskall');
                set(hh, 'AlphaData', 0.5-0.5*isnan(maskall'))
        
                %set the colors for each block using RGB values
                map = [0.5,0,0.4;
                       1,0,0;
                       0,0,1;];
                colormap(map)

            case 'phigh'

            blocp = f.blockp;
            if reg_idx == 0
                pblocks = zeros(nx,ny,3);
                for j = 1:nx
                    for k = 1:ny
                        pblocks(j,k,:) = softmax(squeeze(blocp(j,k,1,:)));
                    end
                end
            end
            shading interp
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),pblocks(:,:,2)','edgecolor','none','facealpha',0.4)
            colorbar


            case 'pi' % probability of opting out. use only with reg_idx=1
            shading interp
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),log(f.pi(:,:,zuse_m)'),'edgecolor','none','facealpha',0.4)
            shading interp
            clim([-0.5,0]);
            colorbar

            otherwise  % do magnitude of dynamics
            colormap(parula)
            shading interp
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),f.sgrad_norm(:,:,zuse_m)','edgecolor','none','facealpha',0.4)
            shading interp

            %set the color range, based on reg_idx
            if reg_idx == 0
                clim([0,1000])
            else
                clim([0,5000])
            end
            colorbar
        end
    
        % set viewing limits for X,Y dims
        xlim([xvec(1), xvec(end)])
        ylim([yvec(1), yvec(end)])
    
        % add labels, titles
        xlabel('PC 1')
        ylabel('PC 2')
        set(gca,'ydir','normal')  % make sure (0,0) is bottom left corner
        hold on      
        title('2d')

        %return x,y,z, to lower res
        xvec = f.X0(1:res1:end); %PC2
        yvec = f.Y0(1:res2:end); % PC1. 
        zvec = f.Z0(1:res3:end);
        nx = length(xvec);
        ny = length(yvec);
        zuse_m = zuse_idx(m);
        [X,Y] = meshgrid(xvec,yvec);
        
        
        %create the flow field arrovs
        dir1 = f.sgrad_x(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m))); % pc1 gradient direction
        dir2 = f.sgrad_y(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m))); % pc2 gradient direction
        dir3 = f.sgrad_z(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m)));
        U = dir1(1:res1:end, 1:res2:end)';
        V = dir2(1:res1:end, 1:res2:end)';
        W = dir3(1:res1:end, 1:res2:end)';

        %decide on color type
        switch bgtype
            case 'block'
                quiver(X,Y,U,V,1, 'color','white','linewidth',2)
            case 'noquiver'

            otherwise
                quiver3(X,Y,zvec(zuse_m)*ones(size(Y)),U,V,W,1,'linewidth',2, 'color',[0.5,0.5,0.5])
        end
        
    end 
end