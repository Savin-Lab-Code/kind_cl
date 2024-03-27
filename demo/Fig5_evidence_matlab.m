% plotting the flow fields, and incongruent trials 
dytype = 'full_cl';


isgen = false;

num = 33;
reg_idx = 0;
idx = 10;
epoch = 'wait';
stage = 'block';
blocktype = 'mixed';
pref = '';

iv = [1,1,1];
res = [3,3,2];

%samples to use? need to dial in
usevec = [2,2,3];  % for ofc

fnum = 17;
zuse_idx = 1;

[f,g, beh, pi_decoder, ~, usepi] = call(dytype, num,reg_idx,idx,epoch,stage, blocktype, isgen, pref);
usepblock = false;
bgtype = 'pmix';


[h] = genflowfield(fnum, f,g,reg_idx, res, bgtype, zuse_idx);
colormap(bone)
xlim([-760.0000 , 570.0109]);
ylim([-858.2223,  465.4004]);


%% add the pmixed lines

xvec = f.X0; %PC2
yvec = f.Y0; % PC1. 
zvec = f.Z0;

nx = length(xvec);
ny = length(yvec);
nz = length(zvec);

blocp = f.blockp;
if reg_idx == 0
    pblocks = zeros(nx,ny,3);
    for j = 1:nx
        for k = 1:ny
            pblocks(j,k,:) = softmax(squeeze(blocp(j,k,1,:)));
        end
    end
end


maskmixed = pblocks(:,:,1) > 0.5;
diffmask = nan(size(maskmixed));
pts = [];
for j = 1:nx
    diffmask = diff(maskmixed(j,:));
    pts_j = find(diffmask ~= 0);
    for k = 1:numel(pts_j)
        ptk = [f.X0(j), f.Y0(pts_j(k))];
        pts = [pts; ptk];
    end
end

scatter(pts(:,1),pts(:,2), 40,'|','k','linewidth',1)


%% Fig. 5 B
datadir = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/dynamics/';
g = load(strcat(datadir,'trialdata_rnn33_incongruent_m2l.mat'));

c_old = [56,83,163]/256;
np = numel(g.trialdata(:,2));

b1 = linspace(0,c_old(1),np);
b2 = linspace(0,c_old(2),np);
b3 = linspace(0,c_old(3),np);
cvec = [b1;b2;b3]';

g.titi(end) = g.titi(end)-1;

scatter(g.trialdata(:,1), g.trialdata(:,2),50,cvec,'filled')
hold on

scatter(g.trialdata(g.tstarts+1,1), g.trialdata(g.tstarts+1,2),50,'green','filled')
scatter(g.trialdata(g.titi+1,1), g.trialdata(g.titi+1,2),50,'red','filled')

%% Fig. 5 supp. high to mixed

datadir = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/dynamics/';
g = load(strcat(datadir,'trialdata_rnn33_incongruent_h2m.mat'));

c_old = [187, 101, 154]/256;
np = numel(g.trialdata(:,2));

b1 = linspace(0,c_old(1),np);
b2 = linspace(0,c_old(2),np);
b3 = linspace(0,c_old(3),np);
cvec = [b1;b2;b3]';

g.titi(end) = g.titi(end)-1;

scatter(g.trialdata(:,1), g.trialdata(:,2),50,cvec,'filled')
scatter(g.trialdata(g.tstarts+1,1), g.trialdata(g.tstarts+1,2),50,'green','filled')
scatter(g.trialdata(g.titi+1,1), g.trialdata(g.titi+1,2),50,'red','filled')



%% Fig 5 supp
datadir = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/dynamics/';
g = load(strcat(datadir,'trialdata_rnn33_incongruent_m2h.mat'));

c_old = [237, 32, 36]/256;
np = numel(g.trialdata(:,2));

b1 = linspace(0,c_old(1),np);
b2 = linspace(0,c_old(2),np);
b3 = linspace(0,c_old(3),np);
cvec = [b1;b2;b3]';

g.titi(end) = g.titi(end)-1;

scatter(g.trialdata(:,1), g.trialdata(:,2),50,cvec,'filled')
scatter(g.trialdata(g.tstarts+1,1), g.trialdata(g.tstarts+1,2),50,'green','filled')
scatter(g.trialdata(g.titi+1,1), g.trialdata(g.titi+1,2),50,'red','filled')




%% Fig 5 supp
datadir = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/dynamics/';
g = load(strcat(datadir,'trialdata_rnn33_incongruent_l2m.mat'));


c_old = [187, 101, 154]/256;
np = numel(g.trialdata(:,2));

b1 = linspace(0,c_old(1),np);
b2 = linspace(0,c_old(2),np);
b3 = linspace(0,c_old(3),np);
cvec = [b1;b2;b3]';

g.titi(end) = g.titi(end)-1;

scatter(g.trialdata(:,1), g.trialdata(:,2),50,cvec,'filled')
scatter(g.trialdata(g.tstarts+1,1), g.trialdata(g.tstarts+1,2),50,'green','filled')
scatter(g.trialdata(g.titi+1,1), g.trialdata(g.titi+1,2),50,'red','filled')


%% draw the colored lines to denote progress from one block to next

figure(265)
clf
hold on

c_old = [56,83,163]/256;
np = numel(g.trialdata(:,2));

b1 = linspace(0,c_old(1),np);
b2 = linspace(0,c_old(2),np);
b3 = linspace(0,c_old(3),np);
cvec = [b1;b2;b3]';

tx = 1:numel(g.trialdata(:,1));

scatter(tx,ones(size(tx)),50,cvec,'filled')

c_old = [187, 101, 154]/256;
np = numel(g.trialdata(:,2));

b1 = linspace(0,c_old(1),np);
b2 = linspace(0,c_old(2),np);
b3 = linspace(0,c_old(3),np);
cvec = [b1;b2;b3]';

scatter(tx,2*ones(size(tx)),50,cvec, 'filled')

c_old = [237, 32, 36]/256;
np = numel(g.trialdata(:,2));

b1 = linspace(0,c_old(1),np);
b2 = linspace(0,c_old(2),np);
b3 = linspace(0,c_old(3),np);
cvec = [b1;b2;b3]';

scatter(tx,3*ones(size(tx)),50,cvec,'filled')





%% function calls

function [f,g,beh, pidecoder, usepblock, usepi] = call(dytype,num,reg_idx,idx,epoch,stage,blocktype, isgen, pref,eps)

    if reg_idx == 0
        usepi = false;
        usepblock = true;
        blocktype = 'mixed';
    else
        usepi = true;
        usepblock = false;
        
    end
    
    switch dytype
        case 'full_cl'      
            dbase = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/';
            datadir = strcat(dbase,'dynamics/flows/');
            datadir_KE = strcat(dbase,'/dynamics/KEmin_constrained/');
        case 'full_cl_test'
            %datadir = '/Users/dhocker/projects/dynamics/results/rnn/ac/20230206_fullclstudy/full_cl/dynamics/';
            
            dbase = '/Users/dhocker/projects/dynamics/results/20231003/full_cl/';
            datadir = strcat(dbase,'flows_end/');
            switch eps
                case 0.1
                    et = 'eps1/';
                case 0.01
                    et = 'eps01/';
                case 0.001
                    et = 'eps001/';
                case 0.05
                    et = 'eps05/';
                otherwise
                    et = 'eps01/';
            end
            datadir_KE = strcat(dbase,'/ke_eps/',et);
  
    
        case 'nok_cl'
            datadir = '/Users/dhocker/projects/dynamics/results/20231003/nok_cl/dynamics/flows/';
            datadir_KE = '/Users/dhocker/projects/dynamics/results/20230206_clstudy/nok_cl/dynamics/KEmin_constrained/';
            
         
            
    end

    if reg_idx == 0
        if isgen
            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_0_mixed_',epoch,'.mat');
        else

            datname = strcat(datadir,'flowfields_rnn_curric_',num2str(num),'_',stage,'_',num2str(idx),'reg_0_mixed_',epoch,'_boutique',pref,'.mat');
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
        g = load(datname_KE);
    end

    beh = f;
    pidecoder = f; % TODO
    

    %rename the samples, because there was a mistake
    %incorrect ordering: mixed, high, low, 5, 10, 20,40,80,...
    %correct ordering  :  5     10,   20, 40 ,80,  M, H, L


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


function [h] = genflowfield(fnum, f,~, reg_idx, res, bgtype, zuse_idx)
    % will make the figure, then output the handle

    res1 = res(1);
    res2 = res(2);
    res3 = res(3);
    
    % set up the background amplitude
    % PC 1 is first component
    % PC2 is second component
    % assumes data is in form of D(y,x) for imagesc, which always confuses me
    
    h = figure(fnum);
    clf
    hold on

    xvec = f.X0; %PC2
    yvec = f.Y0; % PC1. 
    zvec = f.Z0;
    
    nx = length(xvec);
    ny = length(yvec);
    
    for m = 1:numel(zuse_idx)
        zuse_m = zuse_idx(m);
        [X,Y] = meshgrid(xvec,yvec);
    
        %run the softmax
        switch bgtype
            case 'block'
                %blocp = f.blockp(1:res1:end,1:res2:end,1,:);
                blocp = f.blockp;
                if reg_idx == 0
                    pblocks = zeros(nx,ny,3);
                    for j = 1:nx
                        for k = 1:ny
                            pblocks(j,k,:) = softmax(squeeze(blocp(j,k,1,:)));
                        end
                    end
                end
        
                %try nan-out unwanted bits
                maskmixed = 0*ones(size(pblocks(:,:,1)));
                maskmixed(pblocks(:,:,1) < pblocks(:,:,2) + pblocks(:,:,3)) = nan;
        
                maskhigh = ones(size(pblocks(:,:,1)));
                maskhigh(pblocks(:,:,2) < pblocks(:,:,1) + pblocks(:,:,3)) = nan;
        
                masklow = 2*ones(size(pblocks(:,:,2)));
                masklow(pblocks(:,:,3) < pblocks(:,:,1) + pblocks(:,:,2)) = nan;
        
                maskall = nan*ones(3,size(maskmixed,1),size(maskmixed,2));
                maskall(1,:,:) = maskmixed;
                maskall(2,:,:) = maskhigh;
                maskall(3,:,:) = masklow;
        
                maskall = squeeze(sum(maskall,1,'omitnan'));
        
                
                shading interp
                hh = imagesc([xvec(1), xvec(end)], [yvec(1), yvec(end)],maskall');  % trying to plot 
                set(hh, 'AlphaData', 0.5-0.5*isnan(maskall'))

                map = [0.5,0,0.4;
                       1,0,0;
                       0,0,1;];
                colormap(map)

        case 'pi'
            shading interp
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),log(f.pi(:,:,zuse_m)'),'edgecolor','none','facealpha',0.4)
            shading interp
            clim([-0.5,0]);

        otherwise
            colormap(parula)
            shading interp
            surf(X,Y,zvec(zuse_m)*ones(size(Y)),f.sgrad_norm(:,:,zuse_m)','edgecolor','none','facealpha',0.4)
            shading interp
            if reg_idx == 0
                clim([0,1000])
                disp('reg idx == 0')
            else
                clim([0,5000])
            end
        end
    
        xlim([xvec(1), xvec(end)])
        ylim([yvec(1), yvec(end)])
    
        xlabel('PC 1')
        ylabel('PC 2')
        set(gca,'ydir','normal')  % zero is now bottom left corner, 
        hold on
        colorbar
        title('2d')

        %return x,y,z, to lower res
        xvec = f.X0(1:res1:end); %PC2
        yvec = f.Y0(1:res2:end); % PC1. 
        zvec = f.Z0(1:res3:end);
        nx = length(xvec);
        ny = length(yvec);
        zuse_m = zuse_idx(m);
        [X,Y] = meshgrid(xvec,yvec);         
    
        dir1 = f.sgrad_x(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m))); % pc1 gradient direction
        dir2 = f.sgrad_y(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m))); % pc2 gradient direction
        dir3 = f.sgrad_z(:,:,zuse_m)./(sqrt(2*f.sgrad_norm(:,:,zuse_m)));
    
        U = dir1(1:res1:end, 1:res2:end)';
        V = dir2(1:res1:end, 1:res2:end)';
        W = dir3(1:res1:end, 1:res2:end)';

        switch bgtype
            case 'block'
                quiver(X,Y,U,V,1, 'color','white','linewidth',2)
            otherwise
                quiver3(X,Y,zvec(zuse_m)*ones(size(Y)),U,V,W,1,'linewidth',2, 'color',[0.5,0.5,0.5])
        end
        
    end 
end